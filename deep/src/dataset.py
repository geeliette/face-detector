# src/dataset.py
import os, glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

def _collect_pairs(root, mapping):
    files = []
    for sub, label in mapping.items():
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            files += [(f, label) for f in sorted(glob.glob(os.path.join(d, "*.pgm")))]
    return files

class PgmFaces(Dataset):
    """
    Accepts one of these layouts under `root/`:
      - pos/, neg/
      - 1/, 0/
    By default: label 1 = FACE, label 0 = NON-FACE
    """
    def __init__(self, root, files=None, aug=False, face_label=1):
        # Try both conventions
        pairs = []
        if files is None:
            # Convention A: pos/neg
            pairs = _collect_pairs(root, {"pos": face_label, "neg": 1 - face_label})
            # Convention B: 1/0
            if not pairs:
                pairs = _collect_pairs(root, {"1": face_label, "0": 1 - face_label})
        else:
            pairs = files

        if not pairs:
            raise RuntimeError(f"No PGM files found under {root} with expected folders.")

        self.files = pairs

        t = [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        if aug:
            t.insert(0, transforms.RandomApply([
                transforms.RandomAffine(degrees=5, translate=(0.05,0.05), scale=(0.95,1.05))
            ], p=0.5))
        self.tf = transforms.Compose(t)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        f, y = self.files[idx]
        img = Image.open(f)
        img = self.tf(img)
        return img, y

def split_train_val(root, val_ratio=0.1, seed=42, face_label=1):
    # Works for either layout (pos/neg or 1/0)
    pos = []
    neg = []
    for sub, lab in (("pos",face_label), ("neg",1-face_label), ("1",face_label), ("0",1-face_label)):
        d = os.path.join(root, sub)
        if os.path.isdir(d):
            files = sorted(glob.glob(os.path.join(d, "*.pgm")))
            (pos if lab==face_label else neg).extend([(f, lab) for f in files])

    if not pos or not neg:
        raise RuntimeError(f"Could not find both classes under {root}")

    rng = np.random.RandomState(seed)
    rng.shuffle(pos); rng.shuffle(neg)
    npos_val = int(len(pos)*val_ratio)
    nneg_val = int(len(neg)*val_ratio)
    val = pos[:npos_val] + neg[:nneg_val]
    train = pos[npos_val:] + neg[nneg_val:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val
