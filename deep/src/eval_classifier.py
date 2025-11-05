import argparse, torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from .dataset import PgmFaces
from .model import make_model

@torch.no_grad()
def collect_scores(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x,y in loader:
        x = x.to(device)
        logits = model(x)
        prob_face = logits.softmax(1)[:,1].cpu().numpy()
        ys.extend(y)
        ps.extend(prob_face)
    return ys, ps

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="data/test_images")
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--bs", type=int, default=512)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = PgmFaces(args.root, aug=False)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=2)
    model = make_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    y, p = collect_scores(model, dl, device)
    fpr, tpr, thr = roc_curve(y, p)
    print("AUC =", auc(fpr,tpr))

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc(fpr,tpr):.3f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.title("Patch ROC (face vs non-face)")
    plt.show()
