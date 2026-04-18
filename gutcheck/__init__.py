APPROACHES = ["sam31_zs", "sam_lora", "pranet", "dinov3"]

COLORS = {
    "sam31_zs": (0, 120, 255),
    "sam_zs": (60, 160, 240),
    "sam_lora": (160, 80, 255),
    "pranet": (255, 140, 0),
    "dinov3": (60, 200, 120),
    "ground_truth": (255, 255, 255),
}

LABELS = {
    "sam31_zs": "SAM 3.1 zero-shot",
    "sam_zs": "SAM 3 zero-shot",
    "sam_lora": "SAM 3 + LoRA",
    "pranet": "PraNet (2020)",
    "dinov3": "DINOv3 + decoder",
    "ground_truth": "Ground truth",
}
