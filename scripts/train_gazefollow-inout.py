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


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="gazeloom_dinov3_vitl16_inout")
parser.add_argument('--data_path', type=str, default='./data/gazefollow')
parser.add_argument('--ckpt_save_dir', type=str, default='./experiments')
parser.add_argument('--exp_name', type=str, default='train_gazefollow_inout')
parser.add_argument('--log_iter', type=int, default=10)
parser.add_argument('--max_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--n_workers', type=int, default=16)
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint 路径，用于断点续传')
parser.add_argument('--unfreeze_layers', type=int, default=0,
                    help='解冻 backbone 最后 N 层 (默认=0 全冻结)')
args = parser.parse_args()


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


def main():
    exp_dir = os.path.join(args.ckpt_save_dir, args.exp_name,
                           datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(exp_dir, exist_ok=True)
    logger = setup_logging(exp_dir)

    model, transform = get_gazelle_model(args.model)
    model.cuda()

    # ====== 冻结 & 可选解冻 ======
    for param in model.backbone.parameters():
        param.requires_grad = False

    if args.unfreeze_layers > 0:
        logger.info(f"解冻 backbone 的最后 {args.unfreeze_layers} 层")
        for name, param in model.backbone.named_parameters():
            if "blocks" in name:
                try:
                    block_idx = int(name.split('.')[1])
                    if block_idx >= (model.backbone.num_blocks - args.unfreeze_layers):
                        param.requires_grad = True
                except:
                    continue

    # 参数分组
    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith("backbone"):
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": 1e-5},  # backbone 小 lr
        {"params": head_params, "lr": args.lr}    # 其余正常 lr
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-7)

    logger.info(f"可学习参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 数据集
    train_dataset = GazeDataset('gazefollow', args.data_path, 'train', transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                           collate_fn=collate_fn, num_workers=args.n_workers)

    eval_dataset = GazeDataset('gazefollow', args.data_path, 'test', transform)
    eval_dl = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False,
                                          collate_fn=collate_fn, num_workers=args.n_workers)

    bce_loss = nn.BCELoss()  # heatmap
    inout_loss_fn = nn.BCEWithLogitsLoss()  # in/out

    best_min_l2, best_epoch = 1.0, None
    start_epoch = 0

    # ---------- 断点续传 ----------
    if args.resume and os.path.isfile(args.resume):
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_min_l2 = checkpoint.get('best_min_l2', best_min_l2)
        best_epoch = checkpoint.get('best_epoch', best_epoch)

    # ---------- 训练循环 ----------
    for epoch in range(start_epoch, args.max_epochs):
        model.train()
        for cur_iter, batch in enumerate(train_dl):
            # 解包 batch
            imgs, depth_imgs, bbox_norms, gazex, gazey, inout, heights, widths, heatmaps = batch

            optimizer.zero_grad()
            preds = model({"images": imgs.cuda(), "bboxes": bbox_norms})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(1)
            loss_heat = bce_loss(heatmap_preds, heatmaps.cuda())

            if preds['inout'] is not None:
                inout_logits = torch.cat(preds['inout'], dim=0).float()
                inout_gt = inout.cuda().float().view(-1)
                loss_inout = inout_loss_fn(inout_logits, inout_gt)
                loss = loss_heat + 0.5 * loss_inout
            else:
                loss_inout = torch.tensor(0.0)
                loss = loss_heat

            loss.backward()
            optimizer.step()

            if cur_iter % args.log_iter == 0:
                logger.info(f"TRAIN EPOCH {epoch}, iter {cur_iter}/{len(train_dl)}, "
                            f"loss_heat={loss_heat.item():.4f}, loss_inout={loss_inout.item():.4f}, "
                            f"loss_total={loss.item():.4f}")

        scheduler.step()

        # ---------- 保存 checkpoint ----------
        ckpt_path = os.path.join(exp_dir, f'epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_min_l2': best_min_l2,
            'best_epoch': best_epoch
        }, ckpt_path)
        logger.info(f"Saved checkpoint to {ckpt_path}")

        # ---------- eval ----------
        model.eval()
        avg_l2s, min_l2s, aucs, inout_accs = [], [], [], []
        for batch in eval_dl:
            imgs, bboxes, gazex, gazey, inout, heights, widths = batch
            with torch.no_grad():
                preds = model({"images": imgs.cuda(), "bboxes": bboxes})

            heatmap_preds = torch.stack(preds['heatmap']).squeeze(1)
            for i in range(heatmap_preds.shape[0]):
                auc = gazefollow_auc(heatmap_preds[i], gazex[i], gazey[i], heights[i], widths[i])
                avg_l2, min_l2 = gazefollow_l2(heatmap_preds[i], gazex[i], gazey[i])
                aucs.append(auc)
                avg_l2s.append(avg_l2)
                min_l2s.append(min_l2)

            if preds['inout'] is not None:
                inout_logits = torch.cat(preds['inout'], dim=0)
                inout_probs = torch.sigmoid(inout_logits)
                pred_labels = (inout_probs > 0.5).long()
                acc = (pred_labels.cpu() == inout.long().view(-1)).float().mean().item()
                inout_accs.append(acc)

        epoch_auc = np.mean(aucs)
        epoch_min_l2 = np.mean(min_l2s)
        epoch_avg_l2 = np.mean(avg_l2s)
        epoch_inout_acc = np.mean(inout_accs) if inout_accs else None

        logger.info(f"EVAL EPOCH {epoch}: AUC={epoch_auc:.4f}, "
                    f"Min L2={epoch_min_l2:.4f}, Avg L2={epoch_avg_l2:.4f}, "
                    f"InOut Acc={epoch_inout_acc:.4f}" if epoch_inout_acc else "")

        if epoch_min_l2 < best_min_l2:
            best_min_l2 = epoch_min_l2
            best_epoch = epoch

    logger.info(f"Completed training. Best Min L2={best_min_l2:.4f} at epoch {best_epoch}")


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    main()
