"""SAM-Line: a small line-drawing head on top of a frozen SAM 3 vision encoder.

    object crop (RGB + mask) ──────────────────────────────┐
         └─> SAM 3 ViT @1008 ─> stride-7 FPN features ─────┤  LineDecoder (2.3M, full-resolution U-Net)
                              └─> mask-pooled stride-14 ───┘  ─> grey ink drawing of the object

The head sees one frame at a time. It was distilled from a one-step line-art teacher and trained on pseudo-videos
(random camera paths, lighting drift, noise, mask wobble) with a flicker loss, so it answers the same way when the
camera moves.
"""
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAM_SIZE = 1008
DEFAULT_REPO = "Vision-Decision/sam3-line"


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * channels)

    def forward(self, x, cond):
        g, b = self.proj(cond)[:, :, None, None].chunk(2, 1)
        return x * (1 + g) + b


class LineDecoder(nn.Module):
    def __init__(self, sam_dim=256, query_dim=256, widths=(32, 64, 128, 256), n_styles=2, cond_dim=256, out_ch=1):
        super().__init__()
        self.style = nn.Embedding(n_styles, cond_dim)
        self.query_proj = nn.Linear(query_dim, cond_dim)
        self.enc = nn.ModuleList()
        cin = 4  # RGB + mask
        for w in widths:
            self.enc.append(conv_block(cin, w)); cin = w
        self.sam_proj = nn.Conv2d(sam_dim, widths[-1], 1)
        self.dec, self.film = nn.ModuleList(), nn.ModuleList()
        for w_skip, w_in in zip(reversed(widths[:-1]), reversed(widths[1:])):
            self.dec.append(conv_block(w_in + w_skip, w_skip))
            self.film.append(FiLM(cond_dim, w_skip))
        self.bottleneck_film = FiLM(cond_dim, widths[-1])
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, rgb, mask, sam_feat, query, style):
        """rgb (B,3,H,W) in [0,1]; mask (B,1,H,W); sam_feat (B,256,h,w); query (B,256); style (B,) long.
        Returns logits (B,1,H,W): sigmoid = paper brightness (1 = white paper, 0 = ink)."""
        cond = self.style(style) + self.query_proj(query)
        x, skips = torch.cat([rgb, mask], 1), []
        for i, blk in enumerate(self.enc):
            if i:
                x = F.max_pool2d(x, 2)
            x = blk(x); skips.append(x)
        x = x + self.sam_proj(F.interpolate(sam_feat.float(), size=x.shape[-2:], mode="bilinear", align_corners=False))
        x = self.bottleneck_film(x, cond)
        for blk, film, skip in zip(self.dec, self.film, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = film(blk(torch.cat([x, skip], 1)), cond)
        return self.head(x)


def crop_box(mask, margin):
    """(x0, y0, x1, y1) of the mask's bounding box grown by `margin` x its size, clipped to the image."""
    H, W = mask.shape
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int((y1 - y0) * margin), int((x1 - x0) * margin)
    return max(x0 - mx, 0), max(y0 - my, 0), min(x1 + mx, W), min(y1 + my, H)


class SamLine(nn.Module):
    """Frozen SAM 3 vision encoder + the line head. `sam` is a transformers Sam3Model (or anything with a
    `vision_encoder`); load both with `SamLine.from_pretrained()`."""

    def __init__(self, sam, decoder, config):
        super().__init__()
        self.sam, self.decoder, self.config = sam.eval().requires_grad_(False), decoder.eval(), config
        self.size = config["crop_size"]

    @classmethod
    def from_pretrained(cls, repo=DEFAULT_REPO, device="cuda", sam=None, dtype=torch.bfloat16):
        """Download the head from the Hugging Face Hub (or read a local folder) and SAM 3 from facebook/sam3.
        Pass an already-loaded Sam3Model / Sam3VideoModel as `sam` to share its weights."""
        import os
        from safetensors.torch import load_file
        if os.path.isdir(repo):
            cfg_path, w_path = os.path.join(repo, "config.json"), os.path.join(repo, "model.safetensors")
        else:
            from huggingface_hub import hf_hub_download
            cfg_path, w_path = hf_hub_download(repo, "config.json"), hf_hub_download(repo, "model.safetensors")
        config = json.load(open(cfg_path))
        decoder = LineDecoder(**config["decoder"])
        decoder.load_state_dict(load_file(w_path))
        if sam is None:
            from transformers import Sam3Model
            sam = Sam3Model.from_pretrained(config["sam3"]["repo"], revision=config["sam3"].get("revision"),
                                            torch_dtype=dtype)
        return cls(sam, decoder, config).to(device)

    @property
    def device(self):
        return self.decoder.head.weight.device

    @torch.no_grad()
    def features(self, rgb, mask):
        """SAM 3 features for crops rgb (B,3,S,S) in [0,1]: stride-7 map and the mask-pooled stride-14 query."""
        x = F.interpolate(rgb, size=(SAM_SIZE, SAM_SIZE), mode="bilinear", align_corners=False)
        dtype = next(self.sam.vision_encoder.parameters()).dtype
        fpn = self.sam.vision_encoder((x * 2 - 1).to(dtype)).fpn_hidden_states
        feat, sem = fpn[1].float(), fpn[2].float()
        m = F.interpolate(mask, size=sem.shape[-2:], mode="area")
        return feat, (sem * m).sum((2, 3)) / m.sum((2, 3)).clamp_min(1e-3)

    @torch.no_grad()
    def draw_crops(self, items, batch=16):
        """items: list of (rgb uint8 (h,w,3), mask bool (h,w)) object crops. Returns a list of ink maps
        (h,w) float32 in [0,1] (1 = black ink), zero away from the object."""
        S, dev, out = self.size, self.device, []
        g = max(3, round(self.config["mask_dilate"] * S))
        for b in range(0, len(items), batch):
            chunk = items[b:b + batch]
            x = torch.zeros(len(chunk), 3, S, S, device=dev); m = torch.zeros(len(chunk), 1, S, S, device=dev)
            sizes = []
            for k, (rgb, mask) in enumerate(chunk):
                h, w = mask.shape
                s = S / max(h, w)
                hh, ww = max(8, round(h * s)), max(8, round(w * s))
                r = torch.from_numpy(np.ascontiguousarray(rgb)).to(dev).permute(2, 0, 1)[None].float() / 255
                x[k, :, :hh, :ww] = F.interpolate(r, size=(hh, ww), mode="area")[0]
                mm = torch.from_numpy(mask.astype(np.float32)).to(dev)[None, None]
                m[k, :, :hh, :ww] = (F.interpolate(mm, size=(hh, ww), mode="area")[0] > 0.5).float()
                sizes.append((h, w, hh, ww))
            feat, query = self.features(x, m)
            style = torch.full((len(chunk),), self.config["style"], dtype=torch.long, device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = self.decoder(x, m, feat, query, style).float()
            near = F.max_pool2d(m, 2 * g + 1, stride=1, padding=g)
            ink = (1 - torch.sigmoid(logit)) * near
            for k, (h, w, hh, ww) in enumerate(sizes):
                out.append(F.interpolate(ink[k:k + 1, :, :hh, :ww], size=(h, w), mode="bilinear",
                                         align_corners=False)[0, 0].clamp(0, 1).cpu().numpy())
        return out

    @torch.no_grad()
    def draw(self, image, masks, batch=16):
        """image uint8 (H,W,3); masks: list of bool (H,W). Returns the ink map (H,W) of all objects together."""
        H, W = image.shape[:2]
        ink = np.zeros((H, W), np.float32)
        masks = [m for m in masks if m.any()]
        boxes = [crop_box(m, self.config["crop_margin"]) for m in masks]
        items = [(image[y0:y1, x0:x1], m[y0:y1, x0:x1]) for m, (x0, y0, x1, y1) in zip(masks, boxes)]
        for (x0, y0, x1, y1), d in zip(boxes, self.draw_crops(items, batch)):
            ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], d)
        return ink


PAPER = np.array([250, 248, 242], np.float32)
INK = np.array([28, 28, 32], np.float32)


def compose(image, masks, ink, mode="paper"):
    """paper: the objects become paper + ink, the rest of the photo stays. lines: ink on blank paper.
    overlay: ink over the photo."""
    import cv2
    out = image.astype(np.float32)
    union = np.zeros(image.shape[:2], bool)
    for m in masks:
        union |= m
    if mode == "lines":
        out = np.broadcast_to(PAPER, out.shape).copy()
    elif mode == "paper":
        hard = cv2.dilate(union.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)
        fill = np.maximum(hard, cv2.GaussianBlur(hard, (0, 0), 1.0))[..., None]
        out = out * (1 - fill) + PAPER * fill
    a = ink[..., None]
    return (out * (1 - a) + INK * a).clip(0, 255).astype(np.uint8)
