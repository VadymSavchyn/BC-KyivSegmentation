import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROBOFLOW_API_KEY = "PASTE_API_KEY_HERE"
ROBOFLOW_WORKSPACE = "PASTE_WORKSPACE_HERE"
ROBOFLOW_PROJECT = "PASTE_PROJECT_HERE"
ROBOFLOW_VERSION = 1
LOCAL_DATASET_DIR = ""

CONFIG_OUTPUT = Path("/content/dataset_config.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def fetch_dataset_dir():
    if LOCAL_DATASET_DIR:
        dataset_path = Path(LOCAL_DATASET_DIR)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        return dataset_path

    if ROBOFLOW_API_KEY == "PASTE_API_KEY_HERE":
        raise ValueError("Set your Roboflow credentials or specify LOCAL_DATASET_DIR.")

    from roboflow import Roboflow

    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
    dataset = project.version(ROBOFLOW_VERSION).download("yolov11")
    return Path(dataset.location)


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    raw_names = cfg["names"]
    if isinstance(raw_names, dict):
        ordered = [str(raw_names[k]) for k in sorted(raw_names.keys(), key=int)]
    else:
        ordered = [str(name) for name in raw_names]

    return ["background"] + ordered


def parse_yolo_polygons(label_path, width, height):
    if not label_path.exists():
        return []

    polygons = []
    text = label_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue

        cls_id = int(float(parts[0])) + 1
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] = np.clip(np.round(coords[:, 0] * width), 0, width - 1)
        coords[:, 1] = np.clip(np.round(coords[:, 1] * height), 0, height - 1)
        pts = coords.astype(np.int32)
        area = float(cv2.contourArea(pts))
        if area > 0:
            polygons.append((area, cls_id, pts))

    polygons.sort(key=lambda item: item[0], reverse=True)
    return polygons


def convert_split(data_dir, split_name, num_classes):
    img_dir = data_dir / split_name / "images"
    lbl_dir = data_dir / split_name / "labels"
    mask_dir = data_dir / split_name / "masks"

    if not img_dir.exists():
        return 0, np.zeros(num_classes, dtype=np.int64)

    if mask_dir.exists():
        shutil.rmtree(mask_dir)
    mask_dir.mkdir(parents=True, exist_ok=True)

    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    converted = 0

    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        height, width = img.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        for _, cls_id, pts in parse_yolo_polygons(lbl_path, width, height):
            if 0 < cls_id < num_classes:
                cv2.fillPoly(mask, [pts], cls_id)

        cv2.imwrite(str(mask_dir / f"{img_path.stem}.png"), mask)
        counts = np.bincount(mask.ravel(), minlength=num_classes)
        pixel_counts += counts[:num_classes]
        converted += 1

    return converted, pixel_counts


def main():
    data_dir = fetch_dataset_dir()
    class_names = load_class_names(data_dir / "data.yaml")
    num_classes = len(class_names)

    print("Dataset:", data_dir)
    print("Classes:", {idx: name for idx, name in enumerate(class_names)})

    for split_name in ("train", "valid", "test"):
        count, pixel_counts = convert_split(data_dir, split_name, num_classes)
        if count == 0:
            continue

        total_px = int(pixel_counts.sum())
        shares = []
        for idx, name in enumerate(class_names):
            pct = (pixel_counts[idx] / total_px * 100.0) if total_px > 0 else 0.0
            shares.append(f"{name}: {pct:.1f}%")
        print(f"{split_name}: {count} masks | " + ", ".join(shares))

    config_payload = {
        "data_dir": str(data_dir),
        "class_names": class_names,
    }
    CONFIG_OUTPUT.write_text(json.dumps(config_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Saved config:", CONFIG_OUTPUT)


if __name__ == "__main__":
    main()
