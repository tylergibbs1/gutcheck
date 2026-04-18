#!/usr/bin/env bash
# Download Kvasir-SEG, CVC-ClinicDB, and a small HyperKvasir clip subset.
# Idempotent: skips anything already downloaded and extracted.
set -euo pipefail

DATA_DIR="${DATA_DIR:-/workspace/gutcheck/data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "==> Kvasir-SEG"
if [ ! -d Kvasir-SEG ]; then
    if [ ! -f kvasir-seg.zip ]; then
        wget --tries=3 -q --show-progress \
            "https://datasets.simula.no/downloads/kvasir-seg.zip" -O kvasir-seg.zip
    fi
    unzip -q kvasir-seg.zip
    rm kvasir-seg.zip
fi
echo "    images: $(find Kvasir-SEG/images -name '*.jpg' | wc -l)"
echo "    masks : $(find Kvasir-SEG/masks  -name '*.jpg' | wc -l)"

echo "==> CVC-ClinicDB"
if [ ! -d CVC-ClinicDB ]; then
    mkdir -p CVC-ClinicDB
    # Mirror used by most recent polyp-seg benchmarks (PraNet, Polyp-PVT train/test pack).
    # This pack ships CVC-ClinicDB as PNGs under TestDataset/CVC-ClinicDB.
    if [ ! -f polyp_testset.zip ]; then
        wget --tries=3 -q --show-progress \
            "https://github.com/DengPingFan/PraNet/releases/download/v1.0/TestDataset.zip" \
            -O polyp_testset.zip || \
        wget --tries=3 -q --show-progress \
            "https://huggingface.co/datasets/DongfeiJi/Polyp-Segmentation-Datasets/resolve/main/TestDataset.zip" \
            -O polyp_testset.zip
    fi
    unzip -q polyp_testset.zip -d _polyp_tmp
    # The archive contains CVC-ClinicDB/images and CVC-ClinicDB/masks.
    cp -r _polyp_tmp/TestDataset/CVC-ClinicDB/* CVC-ClinicDB/
    rm -rf _polyp_tmp polyp_testset.zip
fi
echo "    images: $(ls CVC-ClinicDB/images | wc -l)"
echo "    masks : $(ls CVC-ClinicDB/masks  | wc -l)"

echo "==> HyperKvasir clips (small subset for video demos)"
mkdir -p HyperKvasir/clips
if [ -z "$(ls HyperKvasir/clips 2>/dev/null)" ]; then
    # Pull a handful of labeled colonoscopy clips from the HyperKvasir labeled-videos set.
    # Use the official Simula mirror; the full labeled-videos tarball is ~32 GB so we
    # grab a small sampler archive if present, otherwise pull individual clips.
    wget --tries=3 -q --show-progress \
        "https://datasets.simula.no/downloads/hyperkvasir/hyper-kvasir-labeled-videos.zip" \
        -O HyperKvasir/labeled-videos.zip || echo "(labeled-videos.zip not available; relying on clip links)"
    if [ -f HyperKvasir/labeled-videos.zip ]; then
        # Extract only a few clips (first 3 files in the archive) to keep disk light.
        cd HyperKvasir
        unzip -l labeled-videos.zip | awk 'NR>3 && /\.(mp4|avi|mov)$/ {print $NF}' | head -3 > _wanted.txt
        if [ -s _wanted.txt ]; then
            unzip -q -j labeled-videos.zip $(cat _wanted.txt) -d clips/ || true
        fi
        rm -f labeled-videos.zip _wanted.txt
        cd ..
    fi
fi
echo "    clips: $(ls HyperKvasir/clips 2>/dev/null | wc -l)"

echo "Done."
