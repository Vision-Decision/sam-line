"""Videos: a text prompt picks the objects, SAM 3 tracks them, the line head draws every frame.

python -m sam_line.video --video in.mp4 --prompt bird --out out/ [--start 0 --duration 10]
    out/paper.mp4   objects as ink drawings on paper, the rest of the frame kept
    out/lines.mp4   only the drawings, on blank paper
    out/side.mp4    original | paper, half size
    out/timing.json

Steadiness comes from four places: the SAM 3 tracker keeps ids, each id's mask logits are averaged over a few
frames, each id's crop box is smoothed in time (crop jitter becomes line jitter), and the ink is blended with
the previous frame's ink moved by optical flow where the flow is forward-backward consistent.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import DEFAULT_REPO, SamLine, compose


def read_frames(src, start=0.0, duration=0.0, max_side=1920):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
           "-of", "json", str(src)]
    st = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)["streams"][0]
    W, H = st["width"], st["height"]
    s = min(1.0, max_side / max(W, H))
    W, H = int(W * s) // 2 * 2, int(H * s) // 2 * 2
    num, den = (int(x) for x in st["r_frame_rate"].split("/"))
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start)] + (["-t", str(duration)] if duration else []) + \
          ["-i", str(src), "-vf", f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3), num / den


def writer(path, W, H, fps):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


@torch.no_grad()
def track(frames, prompt, sam3, dev="cuda"):
    """SAM 3 video tracking from a text prompt -> per frame {object id: mask logits (low res, float16)}."""
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    kw = dict(revision=sam3.get("revision"))
    model = Sam3VideoModel.from_pretrained(sam3["repo"], **kw, torch_dtype=torch.bfloat16).to(dev).eval()
    proc = Sam3VideoProcessor.from_pretrained(sam3["repo"], **kw)
    sess = proc.init_video_session(video=list(frames), inference_device=dev, processing_device=dev,
                                   video_storage_device=dev, dtype=torch.bfloat16)
    proc.add_text_prompt(sess, prompt)
    per = {}
    for out in model.propagate_in_video_iterator(inference_session=sess):
        r = proc.postprocess_outputs(sess, out)
        per[out.frame_idx] = {int(o): out.obj_id_to_mask[o][0].float().cpu().numpy().astype(np.float16)
                              for o in r["object_ids"].tolist()}
    del model, sess
    torch.cuda.empty_cache()
    return [per.get(i, {}) for i in range(len(frames))]


def smooth_masks(logits, H, W, radius=2, min_life=5, min_area=1.5e-4):
    """Centred temporal mean of each id's logits (over frames where the tracker has it), then > 0.
    Ids seen for fewer than `min_life` frames or smaller than `min_area` of the frame are dropped."""
    n = len(logits)
    ids = sorted({o for fr in logits for o in fr})
    life = {o: [i for i in range(n) if o in logits[i]] for o in ids}
    ids = [o for o in ids if len(life[o]) >= min_life
           and np.median([(logits[i][o] > 0).mean() for i in life[o]]) >= min_area]
    masks = []
    for i in range(n):
        fr = {}
        for o in ids:
            if o not in logits[i]:
                continue
            win = [logits[j][o].astype(np.float32) for j in range(max(0, i - radius), min(n, i + radius + 1))
                   if o in logits[j]]
            m = cv2.resize(np.mean(win, 0), (W, H), interpolation=cv2.INTER_LINEAR) > 0
            if m.any():
                fr[o] = m
        masks.append(fr)
    return masks


def stable_boxes(masks, H, W, margin=0.08, center_radius=7, size_radius=15):
    """Per id and frame: crop box (x0, y0, x1, y1) with a smoothed centre and size."""
    raw = {}
    for i, fr in enumerate(masks):
        for o, m in fr.items():
            ys, xs = np.where(m)
            raw.setdefault(o, {})[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    boxes = [dict() for _ in masks]
    for o, bx in raw.items():
        idx = sorted(bx)
        arr = np.array([bx[i] for i in idx], np.float32)
        cx, cy = (arr[:, 0] + arr[:, 2]) / 2, (arr[:, 1] + arr[:, 3]) / 2
        bw, bh = arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1]
        for k, i in enumerate(idx):
            near = [j for j, f in enumerate(idx) if abs(f - i) <= center_radius]
            far = [j for j, f in enumerate(idx) if abs(f - i) <= size_radius]
            c = (cx[near].mean(), cy[near].mean())
            w = min(np.mean([bw[max(0, j - 3):j + 4].max() for j in far]) * (1 + 2 * margin), W)
            h = min(np.mean([bh[max(0, j - 3):j + 4].max() for j in far]) * (1 + 2 * margin), H)
            x0 = int(np.clip(c[0] - w / 2, 0, W - w)); y0 = int(np.clip(c[1] - h / 2, 0, H - h))
            boxes[i][o] = (x0, y0, int(x0 + w), int(y0 + h))
    return boxes


class FlowFilter:
    """z_t = (1 - a) p_t + a warp(z_{t-1}), with a = alpha where the optical flow is forward-backward consistent."""

    def __init__(self, alpha=0.5, scale=0.5):
        self.alpha, self.scale = alpha, scale
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = self.prev = None

    def __call__(self, frame, ink):
        H, W = ink.shape
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), None, fx=self.scale, fy=self.scale)
        z = ink
        if self.prev_gray is not None:
            fw, bw = self.dis.calc(g, self.prev_gray, None), self.dis.calc(self.prev_gray, g, None)
            h, w = g.shape
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            mx, my = gx + fw[..., 0], gy + fw[..., 1]
            back = cv2.remap(bw, mx, my, cv2.INTER_LINEAR)
            ok = (np.hypot(fw[..., 0] + back[..., 0], fw[..., 1] + back[..., 1]) < 1.0).astype(np.float32)
            up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_LINEAR)
            a = self.alpha * up(ok)
            z = (1 - a) * ink + a * cv2.remap(self.prev, up(mx) / self.scale, up(my) / self.scale, cv2.INTER_LINEAR)
        self.prev_gray, self.prev = g, z
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_REPO, help="Hub repo or local folder of the line head")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--max-side", type=int, default=1920)
    ap.add_argument("--chunk", type=int, default=16, help="frames per drawing batch")
    ap.add_argument("--no-filter", action="store_true", help="skip the optical-flow ink filter")
    a = ap.parse_args()
    dev = "cuda"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    T, t0 = {}, time.perf_counter()
    frames, fps = read_frames(a.video, a.start, a.duration, a.max_side)
    n, H, W = frames.shape[:3]
    T["decode_s"] = time.perf_counter() - t0; t = time.perf_counter()
    model = SamLine.from_pretrained(a.model, device=dev)
    T["load_s"] = time.perf_counter() - t; t = time.perf_counter()
    logits = track(frames, a.prompt, model.config["sam3"], dev)
    T["track_s"] = time.perf_counter() - t; t = time.perf_counter()
    masks = smooth_masks(logits, H, W)
    boxes = stable_boxes(masks, H, W, margin=model.config["crop_margin"])
    T["masks_s"] = time.perf_counter() - t
    filt = None if a.no_filter else FlowFilter()
    ws = {k: writer(out / f"{k}.mp4", W, H, fps) for k in ("paper", "lines")}
    hw, hh = W // 4 * 2, H // 4 * 2          # half size, even
    ws["side"] = writer(out / "side.mp4", 2 * hw, hh, fps)
    t_draw = t_rest = 0.0
    for c0 in range(0, n, a.chunk):
        t = time.perf_counter()
        idx = range(c0, min(n, c0 + a.chunk))
        keys = [(i, o) for i in idx for o in boxes[i]]
        items = [(frames[i][y0:y1, x0:x1], masks[i][o][y0:y1, x0:x1])
                 for i, o in keys for (x0, y0, x1, y1) in [boxes[i][o]]]
        drawn = model.draw_crops(items)
        torch.cuda.synchronize(); t_draw += time.perf_counter() - t; t = time.perf_counter()
        ink = {i: np.zeros((H, W), np.float32) for i in idx}
        for (i, o), d in zip(keys, drawn):
            x0, y0, x1, y1 = boxes[i][o]
            ink[i][y0:y1, x0:x1] = np.maximum(ink[i][y0:y1, x0:x1], d)
        for i in idx:
            z = filt(frames[i], ink[i]) if filt is not None else ink[i]
            ms = list(masks[i].values())
            paper = compose(frames[i], ms, z, "paper")
            ws["paper"].stdin.write(paper.tobytes())
            ws["lines"].stdin.write(compose(frames[i], ms, z, "lines").tobytes())
            half = lambda x: cv2.resize(x, (hw, hh), interpolation=cv2.INTER_AREA)
            ws["side"].stdin.write(np.concatenate([half(frames[i]), half(paper)], 1).tobytes())
        t_rest += time.perf_counter() - t
    for w in ws.values():
        w.stdin.close(); w.wait()
    total = time.perf_counter() - t0
    T.update(frames=n, W=W, H=H, fps=round(fps, 3), objects_per_frame=round(sum(map(len, boxes)) / max(n, 1), 2),
             track_ms_per_frame=T["track_s"] / n * 1e3, draw_ms_per_frame=t_draw / n * 1e3,
             filter_compose_write_ms_per_frame=t_rest / n * 1e3, total_s=total, processed_fps=n / total)
    T = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in T.items()}
    json.dump(T, open(out / "timing.json", "w"), indent=1)
    print(T, flush=True)


if __name__ == "__main__":
    main()
