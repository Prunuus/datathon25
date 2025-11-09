"""Train DecisionNet on self-play dataset (CPU-only).

Stronger training settings:
- AdamW optimizer with weight decay
- Cosine LR schedule with warmup
- Label smoothing for directions, BCE for boost
- Gradient clipping and early stopping on val loss (optional split)

Usage:
    python train_model.py --data dataset2.pt --out weights2.pt --epochs 15 --batch 1024 --lr 3e-4
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import DecisionNet
import math


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=str, default='dataset2.pt')
    ap.add_argument('--out', type=str, default='weights2.pt')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--warmup_steps', type=int, default=200)
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--val_split', type=float, default=0.1)
    args = ap.parse_args()

    data = torch.load(args.data, map_location='cpu')
    X = data['X']  # [N, D]
    y_dir = data['y_dir']  # [N]
    y_boost = data['y_boost']  # [N]
    D = int(data['feature_dim'])

    model = DecisionNet(D)
    model.train()
    torch.set_num_threads(1)

    ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    bce = nn.BCEWithLogitsLoss()
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Train/val split
    N = X.shape[0]
    if args.val_split > 0.0 and N > 100:
        n_val = int(N * args.val_split)
        perm_all = torch.randperm(N)
        val_idx = perm_all[:n_val]
        tr_idx = perm_all[n_val:]
        Xtr, Xval = X[tr_idx], X[val_idx]
        ytr_d, yval_d = y_dir[tr_idx], y_dir[val_idx]
        ytr_b, yval_b = y_boost[tr_idx], y_boost[val_idx]
    else:
        Xtr, ytr_d, ytr_b = X, y_dir, y_boost
        Xval, yval_d, yval_b = None, None, None

    Ntr = Xtr.shape[0]
    global_step = 0
    best_val = float('inf')
    best_state = None
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(Ntr)
        total_loss = 0.0
        for i in range(0, Ntr, args.batch):
            idx = perm[i:i+args.batch]
            xb = Xtr[idx]
            yb_dir = ytr_d[idx]
            yb_boost = ytr_b[idx]

            dir_logits, boost_logit = model(xb)
            loss = ce(dir_logits, yb_dir) + 0.1 * bce(boost_logit, yb_boost)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            # cosine schedule with warmup
            global_step += 1
            if args.warmup_steps > 0 and global_step < args.warmup_steps:
                for pg in opt.param_groups:
                    pg['lr'] = args.lr * global_step / args.warmup_steps
            else:
                # cosine decay to 10% of base lr
                progress = (global_step - args.warmup_steps) / max(1, (args.epochs * math.ceil(Ntr/args.batch)) - args.warmup_steps)
                progress = min(max(progress, 0.0), 1.0)
                lr_now = (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (1 - progress)))) * args.lr
                for pg in opt.param_groups:
                    pg['lr'] = lr_now
            total_loss += loss.item() * xb.size(0)
        tr_loss = total_loss / Ntr
        log = f"Epoch {epoch}: train_loss={tr_loss:.4f}"
        # Validation
        if Xval is not None:
            model.eval()
            with torch.no_grad():
                vb = 0.0
                for j in range(0, Xval.shape[0], args.batch):
                    xb = Xval[j:j+args.batch]
                    yb_dir = yval_d[j:j+args.batch]
                    yb_boost = yval_b[j:j+args.batch]
                    dir_logits, boost_logit = model(xb)
                    loss = ce(dir_logits, yb_dir) + 0.1 * bce(boost_logit, yb_boost)
                    vb += loss.item() * xb.size(0)
                val_loss = vb / Xval.shape[0]
                log += f", val_loss={val_loss:.4f}"
                # Early stopping checkpointing
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            model.train()
        print(log)

    state_to_save = best_state if best_state is not None else model.state_dict()
    torch.save(state_to_save, args.out)
    print(f"Saved weights to {args.out}")


if __name__ == '__main__':
    main()
