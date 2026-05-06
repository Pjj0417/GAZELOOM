import argparse
from datetime import datetime
import numpy as np
import os
import random
import torch
import torch.nn as nn
import logging

from gazeloom.dataloader import GazeDataset, collate_fn
from gazeloom.model import get_gazelle_model
from gazeloom.utils import gazefollow_auc, gazefollow_l2


# === Warmup + Cosine Scheduler ===
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_iters, max_iters, min_lr=1e-7, last_epoch=-1):
        self.warmup_iters = warmup_iters
        self.max_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_iters:
            return [base_lr * (step + 1) / self.warmup_iters for base_lr in self.base_lrs]
        progress = (step - self.warmup_iters) / float(self.max_iters - self.warmup_iters)
        return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                for base_lr in self.base_lrs]


# === Args ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="gazeloom_dinov3_vitl16_inout")
    parser.add_argument('--data_path', type=str, default='./data/gazefollow')
    parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
    parser.add_argument('--exp_name', type=str, default='train_gazefollow')
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--max_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--n_workers', type=int, default=16)
    parser.add_argument('--resume', type=str, default=None)

    # loss 权重
    parser.add_argument('--w_heatmap', type=float, default=1.0)
    parser.add_argument('--w_inout', type=float, default=0.2)

    # 解冻控制
    parser.add_argument('--unfreeze_layers', type=int, default=0,
                        help="解冻 DINOv3 最后的 N 层 transformer block (默认 0=全冻结)")

    return parser.parse_args()


# === Logging ===
def setup_logging(exp_dir):
    log_dir = os.path.join(exp_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    return logging.getLogger(__name__)


# === Checkpoint ===
def save_checkpoint(path, epoch, model, optimizer, scheduler, best_min_l2, best_epoch, logger, is_best=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_min_l2': best_min_l2,
        'best_epoch': best_epoch
    }, path)
    if is_best:
        logger.info(f"✅ New best model saved at {path}")
    else:
        logger.info(f"Saved checkpoint to {path}")


# === Main ===
def main():
    args = parse_args()

    # ==== 实验目录 ====
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logging(exp_dir)

    # ==== 模型 ====
    model, transform = get_gazelle_model(args.model)
    model.cuda()

    # 冻结 backbone
    for param in model.backbone.parameters():
        param.requires_grad = False

    # 解冻最后 N 层
    if args.unfreeze_layers > 0 and hasattr(model.backbone.model, "blocks"):
        for block in model.backbone.model.blocks[-args.unfreeze_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        logger.info(f"解冻了 DINOv3 的最后 {args.unfreeze_layers} 层 transformer block")

    # ==== 优化器 / scheduler / 损失 ====
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    warmup_iters = 2 * (113000 // args.batch_size)  # 假设 train 有 ~113k 样本
    max_iters = args.max_epochs * (113000 // args.batch_size)
    scheduler = WarmupCosineLR(optimizer, warmup_iters, max_iters, min_lr=1e-7)

    bce_loss = nn.BCELoss()

    # ==== 数据 ====
    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.n_workers
    )
    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.n_workers
    )

    # ==== 状态初始化 ====
    best_min_l2, best_epoch = 1.0, None
    start_epoch = 0

    # ==== Resume ====
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cuda")

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_min_l2 = checkpoint.get('best_min_l2', best_min_l2)
        best_epoch = checkpoint.get('best_epoch', best_epoch)
        logger.info(f"Resumed from epoch {start_epoch}")

    logger.info(f"可学习参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # ==== Training ====
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            imgs, depth_maps, bboxes, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "depth_map": depth_maps.cuda(),
                           "bboxes": [[bbox] for bbox in bboxes]})

            # heatmap loss
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            loss_heatmap = bce_loss(heatmap_preds, heatmaps.cuda())
            loss = args.w_heatmap * loss_heatmap

            # inout loss
            if preds["inout"] is not None:
                inout_preds = torch.cat(preds["inout"])
                inout_gt = inout.float().cuda()
                loss_inout = bce_loss(inout_preds, inout_gt)
                loss = loss + args.w_inout * loss_inout
            else:
                loss_inout = torch.tensor(0.0)

            loss.backward()
            optimizer.step()
            scheduler.step()

            if cur_iter % args.log_iter == 0:
                logger.info(f"TRAIN EPOCH {epoch}, iter {cur_iter}/{len(train_dl)}, "
                            f"loss={loss.item():.4f}, "
                            f"heatmap={loss_heatmap.item():.4f}, "
                            f"inout={loss_inout.item():.4f}, "
                            f"lr={scheduler.get_last_lr()[0]:.6f}")

        # ==== Save checkpoint ====
        ckpt_path = os.path.join(exp_dir, f'epoch_{epoch}.pt')
        save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, best_min_l2, best_epoch, logger)

        # ==== Validation ====
        model.eval()
        avg_l2s, min_l2s, aucs = [], [], []
        for batch in eval_dl:
            imgs, depth_maps, bboxes, gazex, gazey, inout, heights, widths = batch
            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "depth_map": depth_maps.cuda(),
                               "bboxes": [[bbox] for bbox in bboxes]})
            heatmap_preds = torch.stack(preds['heatmap']).squeeze(dim=1)
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

        epoch_auc = np.mean(aucs)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_avg_l2 = np.mean(avg_l2s)
        logger.info(f"EVAL EPOCH {epoch}: AUC={epoch_auc:.4f}, Min L2={epoch_min_l2:.4f}, Avg L2={epoch_avg_l2:.4f}")

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch
            best_path = os.path.join(exp_dir, "best_model.pt")
            save_checkpoint(best_path, epoch, model, optimizer, scheduler,
                            best_min_l2, best_epoch, logger, is_best=True)

    logger.info(f"Completed training. Best Min L2={best_min_l2:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()
# python train_gazefollow_CGA.py --model  gazelle_cgaf_dinov3_vith16  --data_path ./data/gazefollow  --batch_size 48  --lr 5e-4 --exp_name gazefollow_9_30_CGA
    
# python train_gazefollow_CGA.py --model  gazelle_cgaf_dinov3_vitl16  --data_path ./data/gazefollow  --batch_size 48  --lr 5e-4 --exp_name gazefollow_9_30_CGAl
