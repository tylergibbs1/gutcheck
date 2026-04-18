"""Shared config constants for every script in this folder."""

from __future__ import annotations

from pathlib import Path

ROOT = Path("/workspace/gutcheck")
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"

KVASIR_ROOT = DATA / "Kvasir-SEG"
KVASIR_TEST_SPLIT_DIR = DATA / "TestDataset" / "Kvasir" / "images"
CVC_TEST_ROOT = DATA / "CVC-ClinicDB"

SAM3_DIR = CHECKPOINTS / "sam3"
SAM3_LORA_DIR = CHECKPOINTS / "sam3_lora"
DINOV3_DIR = CHECKPOINTS / "dinov3-vitl16"
DINOV3_HEAD_PATH = CHECKPOINTS / "dinov3_head.pt"
PRANET_REPO = CHECKPOINTS / "PraNet"
PRANET_CKPT = PRANET_REPO / "snapshots" / "PraNet_Res2Net" / "PraNet-19.pth"

METRICS_MASTER = RESULTS / "metrics.json"

TEST_SETS = {
    "kvasir": KVASIR_ROOT,
    "cvc_clinicdb": CVC_TEST_ROOT,
}
