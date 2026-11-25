"""Images: a text prompt picks the objects (SAM 3), the line head draws them.

python -m sam_line.image --image photo.jpg --prompt dog --out out/
    out/<name>_paper.jpg   the objects turned into ink drawings on paper, the rest of the photo kept
    out/<name>_lines.png   only the drawings, on blank paper
    out/<name>_sheet.jpg   photo | paper | lines
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import DEFAULT_REPO, SamLine, compose


class Segmenter:
    """SAM 3 text-prompted instance segmentation on the Sam3Model that SamLine already holds."""

    def __init__(self, sam, sam3=None):
        from transformers import Sam3Processor
        sam3 = sam3 or {"repo": "facebook/sam3"}
        self.sam = sam
        self.proc = Sam3Processor.from_pretrained(sam3["repo"], revision=sam3.get("revision"))

    @torch.no_grad()
    def __call__(self, image, prompt, threshold=0.5):
        dev = next(self.sam.parameters()).device
        dtype = next(self.sam.parameters()).dtype
        inputs = self.proc(images=image, text=prompt, return_tensors="pt").to(dev)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
        out = self.sam(**inputs)
        r = self.proc.post_process_instance_segmentation(out, threshold=threshold, mask_threshold=0.5,
                                                         target_sizes=[image.size[::-1]])[0]
        return [m.cpu().numpy().astype(bool) for m in r["masks"]], r["scores"].float().cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True)
    ap.add_argument("--prompt", required=True, help="what to draw, e.g. 'dog' or 'person'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_REPO, help="Hub repo or local folder of the line head")
    ap.add_argument("--threshold", type=float, default=0.5, help="SAM 3 detection score threshold")
    ap.add_argument("--max-side", type=int, default=2048, help="downscale larger photos so the long side fits")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamLine.from_pretrained(a.model, device=dev)
    seg = Segmenter(model.sam, model.config["sam3"])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for path in a.image:
        image = Image.open(path).convert("RGB")
        image.thumbnail((a.max_side, a.max_side), Image.LANCZOS)
        rgb = np.array(image)
        sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
        sync(); t0 = time.perf_counter()
        masks, scores = seg(image, a.prompt, a.threshold)
        sync(); t1 = time.perf_counter()
        ink = model.draw(rgb, masks)
        sync(); t2 = time.perf_counter()
        name = Path(path).stem
        paper, lines = compose(rgb, masks, ink, "paper"), compose(rgb, masks, ink, "lines")
        Image.fromarray(paper).save(out / f"{name}_paper.jpg", quality=92)
        Image.fromarray(lines).save(out / f"{name}_lines.png")
        Image.fromarray(np.concatenate([rgb, paper, lines], 1)).save(out / f"{name}_sheet.jpg", quality=90)
        print(f"{name}: {len(masks)} x '{a.prompt}' ({rgb.shape[1]}x{rgb.shape[0]})  segment {(t1 - t0) * 1e3:.0f} ms  "
              f"draw {(t2 - t1) * 1e3:.0f} ms", flush=True)


if __name__ == "__main__":
    main()
