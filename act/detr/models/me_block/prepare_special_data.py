from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image, ImageOps


TARGET_WIDTH = 640
TARGET_HEIGHT = 480
VALID_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANIFEST_NAME = "special_data_manifest.json"


def parse_args(default_data_root: str = "") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare special_data images for me_block annotation/training.")
    parser.add_argument("--data-root", type=str, default=default_data_root, help="Root containing special_data.")
    parser.add_argument("--special-dirname", type=str, default="special_data", help="Directory that stores special me_block images.")
    parser.add_argument("--force", action="store_true", help="Rebuild processed rgb images even if already indexed in the manifest.")
    return parser.parse_args()


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.is_file():
        return {"items": []}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(path: Path, manifest: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def crop_to_target(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    target_ratio = TARGET_WIDTH / TARGET_HEIGHT
    current_ratio = width / height

    if current_ratio > target_ratio:
        crop_height = height
        crop_width = int(round(height * target_ratio))
    else:
        crop_width = width
        crop_height = int(round(width / target_ratio))

    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    image = image.crop((left, top, left + crop_width, top + crop_height))
    return image.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)


def list_source_images(special_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in special_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VALID_SUFFIXES
    )


def next_output_index(rgb_dir: Path) -> int:
    existing = [path.stem for path in rgb_dir.glob("*.jpg")]
    if not existing:
        return 0
    return max(int(stem) for stem in existing if stem.isdigit()) + 1


def main(default_data_root: str = "") -> None:
    args = parse_args(default_data_root=default_data_root)
    data_root = Path(args.data_root)
    special_dir = data_root / args.special_dirname
    rgb_dir = special_dir / "rgb"
    label_dir = special_dir / "importance_labels"
    preview_dir = special_dir / "importance_labels_preview"
    manifest_path = special_dir / MANIFEST_NAME

    if not special_dir.is_dir():
        raise FileNotFoundError(f"special_data directory does not exist: {special_dir}")

    rgb_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path)
    indexed_hashes = {item["sha1"]: item for item in manifest.get("items", [])}
    output_index = next_output_index(rgb_dir)

    added = 0
    skipped = 0
    for source_path in list_source_images(special_dir):
        source_hash = sha1_file(source_path)
        if not args.force and source_hash in indexed_hashes:
            skipped += 1
            continue

        output_name = f"{output_index:06d}.jpg"
        output_path = rgb_dir / output_name
        with Image.open(source_path) as image:
            processed = crop_to_target(image)
            processed.save(output_path, format="JPEG", quality=95)

        indexed_hashes[source_hash] = {
            "source_name": source_path.name,
            "sha1": source_hash,
            "rgb_name": output_name,
        }
        output_index += 1
        added += 1
        print(f"Processed {source_path.name} -> rgb/{output_name}")

    manifest["items"] = sorted(indexed_hashes.values(), key=lambda item: item["rgb_name"])
    save_manifest(manifest_path, manifest)

    print(f"special_data ready: {special_dir}")
    print(f"Added {added} image(s), skipped {skipped} already indexed image(s).")


if __name__ == "__main__":
    main()
