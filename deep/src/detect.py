import argparse, math
import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from .model import make_model
from .utils import nms

def image_pyramid(img, scale=1.2, min_size=36):
    """
    Yields (rescaled_image, scale_factor_relative_to_original)
    """
    w, h = img.size
    factor = 1.0
    cur = img
    while min(cur.size) >= min_size:
        yield cur, factor
        factor /= scale
        new_w = int(w * factor)
        new_h = int(h * factor)
        if new_w < min_size or new_h < min_size: break
        cur = img.resize((new_w, new_h), Image.BILINEAR)

def sliding_windows(im_w, im_h, win=36, step=8):
    for y in range(0, im_h - win + 1, step):
        for x in range(0, im_w - win + 1, step):
            yield x, y, x+win, y+win

def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    model = make_model().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((36,36)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    img = Image.open(args.image).convert("RGB")
    boxes, scores = [], []
    win = 36
    for scaled, factor in image_pyramid(img, scale=args.scale, min_size=win):
        w, h = scaled.size
        # dense conv trick (optional): here we do naive sliding-window for clarity
        patches, coords = [], []
        for x1,y1,x2,y2 in sliding_windows(w, h, win=win, step=args.step):
            crop = scaled.crop((x1,y1,x2,y2))
            patches.append(tf(crop))
            coords.append((x1,y1,x2,y2, factor))
        if not patches: continue
        x = torch.stack(patches, 0).to(device)
        with torch.no_grad():
            prob = model(x).softmax(1)[:,1].cpu().numpy()
        # thresholding
        for (x1,y1,x2,y2,f), p in zip(coords, prob):
            if p >= args.thr:
                # map back to original image coords
                inv = 1.0/f
                boxes.append((int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv)))
                scores.append(float(p))

    # NMS
    keep = nms(boxes, scores, iou_thresh=args.nms)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]

    # draw
    out = img.copy()
    dr = ImageDraw.Draw(out)
    for (x1,y1,x2,y2), s in zip(boxes, scores):
        dr.rectangle([x1,y1,x2,y2], outline=(255,0,0), width=2)
        dr.text((x1, max(0,y1-10)), f"{s:.2f}", fill=(255,0,0))
    out.save(args.out)
    print(f"Detections: {len(boxes)} | saved to {args.out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--ckpt", default="checkpoints/best.pt")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--scale", type=float, default=1.2)
    p.add_argument("--step", type=int, default=8)
    p.add_argument("--thr", type=float, default=0.9)
    p.add_argument("--nms", type=float, default=0.3)
    p.add_argument("--out", default="out.jpg")
    args = p.parse_args()
    main(args)
