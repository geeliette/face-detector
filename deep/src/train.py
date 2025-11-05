import os, argparse, time
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from tqdm import tqdm
from .dataset import PgmFaces, split_train_val
from .model import make_model

def train_one_epoch(model, loader, opt, device, criterion):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in tqdm(loader, leave=False):
        x, y = x.to(device), torch.tensor(y).long().to(device)
        opt.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

@torch.no_grad()
def eval_epoch(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x, y = x.to(device), torch.tensor(y).long().to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    train_files, val_files = split_train_val(args.train_root, val_ratio=0.1, seed=42)
    ds_train = PgmFaces(args.train_root, files=train_files, aug=True)
    ds_val   = PgmFaces(args.train_root, files=val_files,   aug=False)
    dl_train = DataLoader(ds_train, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

    model = make_model().to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    patience, bad = 5, 0
    os.makedirs(args.out, exist_ok=True)
    for epoch in range(1, args.epochs+1):
        tr_loss, tr_acc = train_one_epoch(model, dl_train, opt, device, criterion)
        va_loss, va_acc = eval_epoch(model, dl_val, device, criterion)
        print(f"[{epoch}/{args.epochs}] train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_root", default="data/train_images")
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()
    main(args)
