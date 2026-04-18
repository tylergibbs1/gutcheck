"""LoRA fine-tune SAM 3's mask decoder on Kvasir-SEG training split.

Keeps vision encoder, DETR decoder, and text encoder frozen. Optimizes only
the LoRA adapters injected into linear layers under `mask_decoder`.
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gutcheck.data import KvasirSEG
from gutcheck.models.sam_wrapper import attach_lora, sam3_mask_loss
from scripts._common import (
    SAM3_DIR,
    SAM3_LORA_DIR,
    KVASIR_ROOT,
    KVASIR_TEST_SPLIT_DIR,
    LOGS,
)

from transformers import Sam3Model, Sam3Processor


class KvasirTrainSet(Dataset):
    def __init__(self, dataset, processor, prompt: str = "polyp"):
        self.ds = dataset
        self.proc = processor
        self.prompt = prompt
        self._cache_input_size: tuple[int, int] | None = None

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds.get(idx)
        # Let the processor handle resize & normalization for the image.
        enc = self.proc(images=s.image, text=self.prompt, return_tensors="pt")
        # Resize the ground-truth mask to match the processor's canonical input size.
        _, _, h, w = enc["pixel_values"].shape
        gt_lowres = torch.from_numpy(s.mask.astype(np.uint8)).float()
        gt_lowres = F.interpolate(gt_lowres[None, None], size=(h, w), mode="nearest")[0, 0]
        return {
            "pixel_values": enc["pixel_values"][0],
            "input_ids": enc["input_ids"][0],
            "attention_mask": enc["attention_mask"][0],
            "original_sizes": enc["original_sizes"][0],
            "gt": gt_lowres,
        }


def collate(batch: list[dict]) -> dict:
    max_len = max(b["input_ids"].shape[0] for b in batch)
    def pad(t, fill=0):
        pad_amt = max_len - t.shape[0]
        if pad_amt <= 0:
            return t
        return F.pad(t, (0, pad_amt), value=fill)
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([pad(b["input_ids"]) for b in batch]),
        "attention_mask": torch.stack([pad(b["attention_mask"]) for b in batch]),
        "original_sizes": torch.stack([b["original_sizes"] for b in batch]),
        "gt": torch.stack([b["gt"] for b in batch]),
    }


def main():
    seed = 1234
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda")

    processor = Sam3Processor.from_pretrained(str(SAM3_DIR))
    model = Sam3Model.from_pretrained(str(SAM3_DIR), torch_dtype=torch.bfloat16).to(device)
    model.train()

    peft_model, targets = attach_lora(model, r=16, alpha=32, dropout=0.05)
    print("LoRA targets:", targets[:20], "...")

    kvasir = KvasirSEG(KVASIR_ROOT, split="train", held_out_ids_path=KVASIR_TEST_SPLIT_DIR)
    print(f"train n={len(kvasir)}")

    train_ds = KvasirTrainSet(kvasir, processor)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=8, collate_fn=collate, pin_memory=True)

    opt = AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=5e-4, weight_decay=1e-4)

    epochs = 10
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / "train_sam_lora.log"
    log_f = log_path.open("a", buffering=1)

    step = 0
    for ep in range(epochs):
        t0 = time.time()
        running = 0.0
        for batch in tqdm(loader, desc=f"epoch {ep}"):
            pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            ids = batch["input_ids"].to(device)
            am = batch["attention_mask"].to(device)
            gt = batch["gt"].to(device)

            out = peft_model(pixel_values=pv, input_ids=ids, attention_mask=am, multimask_output=False)
            loss = sam3_mask_loss(out, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], 1.0)
            opt.step()

            running += loss.item()
            step += 1
            if step % 25 == 0:
                msg = f"step={step} ep={ep} loss={loss.item():.4f}"
                print(msg); log_f.write(msg + "\n")
        ep_loss = running / max(1, len(loader))
        msg = f"epoch {ep} done loss_avg={ep_loss:.4f} dt={time.time()-t0:.1f}s"
        print(msg); log_f.write(msg + "\n")

    SAM3_LORA_DIR.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(SAM3_LORA_DIR))
    print(f"saved LoRA adapters to {SAM3_LORA_DIR}")
    log_f.close()


if __name__ == "__main__":
    main()
