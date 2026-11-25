import json
import shutil
from pathlib import Path

import albumentations as A
import cv2
import geopandas as gpd
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.validation import make_valid

TILES_DIR = Path("/content/test_tiles")
WEIGHTS_PATH = Path("/content/best_segformer.pth")
SHP_DIR = Path("/content/outputs/predictions_segformer_shp")
ZIP_BASE = Path("/content/outputs/predictions_segformer_shp")
PREVIEW_PATH = Path("/content/previews/segformer_mosaic_preview.png")

MIN_AREA_M2 = 4.0
SIMPLIFY_TOLERANCE_M = 0.0
USE_TTA = True
MAX_PREVIEW_SIDE = 2048
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

PREVIEW_PALETTE_BGR = [
    (30, 30, 30),
    (80, 210, 120),
    (220, 175, 95),
    (195, 110, 215),
    (225, 130, 50),
    (70, 210, 235),
    (90, 90, 235),
]


def find_metadata_file():
    preferred = Path("/content/metadata.json")
    candidates = [preferred] if preferred.exists() else []
    candidates.extend(sorted(Path("/content").glob("*.json")))

    seen = set()
    for path in candidates:
        if path in seen or not path.exists():
            continue
        seen.add(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and "tiles" in data and "crs" in data:
            return path, data

    raise FileNotFoundError("Upload metadata .json with crs and tiles into /content/")


def make_blend_window(height, width, floor=0.05):
    wy = np.hanning(height) if height > 1 else np.ones(1, dtype=np.float32)
    wx = np.hanning(width) if width > 1 else np.ones(1, dtype=np.float32)
    window = np.outer(wy, wx).astype(np.float32)
    return floor + (1.0 - floor) * window


def predict_tile_probs(model, image_rgb, infer_tf, device, use_amp, use_tta):
    orig_h, orig_w = image_rgb.shape[:2]
    tensor = infer_tf(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad(), torch.amp.autocast(device, enabled=use_amp):
        probs = torch.softmax(model(tensor), dim=1)
        if use_tta:
            p_h = torch.flip(torch.softmax(model(torch.flip(tensor, dims=[3])), dim=1), dims=[3])
            p_v = torch.flip(torch.softmax(model(torch.flip(tensor, dims=[2])), dim=1), dims=[2])
            p_hv = torch.flip(torch.softmax(model(torch.flip(tensor, dims=[2, 3])), dim=1), dims=[2, 3])
            probs = (probs + p_h + p_v + p_hv) * 0.25

        if probs.shape[2] != orig_h or probs.shape[3] != orig_w:
            probs = F.interpolate(probs, size=(orig_h, orig_w), mode="bilinear", align_corners=False)

    return probs[0].float().cpu().numpy()


def extract_polygons(geom):
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom] if geom.area > 0 else []
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        parts = []
        for sub in geom.geoms:
            parts.extend(extract_polygons(sub))
        return parts
    return []


def save_preview(pred_mask, valid_mask, preview_path):
    height, width = pred_mask.shape
    scale = min(1.0, MAX_PREVIEW_SIDE / max(height, width))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))

    small_pred = cv2.resize(pred_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    small_valid = cv2.resize(valid_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[small_valid] = PREVIEW_PALETTE_BGR[0]
    fg_colors = PREVIEW_PALETTE_BGR[1:]
    for cls_id in np.unique(small_pred):
        if cls_id == 0:
            continue
        color = fg_colors[(int(cls_id) - 1) % len(fg_colors)]
        canvas[small_pred == cls_id] = color

    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), canvas)

    try:
        from IPython.display import Image, display
        display(Image(filename=str(preview_path), width=min(900, out_w)))
    except Exception:
        pass


def download_in_colab(file_path):
    try:
        from google.colab import files
        files.download(str(file_path))
    except ImportError:
        pass


def main():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Upload weights to {WEIGHTS_PATH}")
    if not TILES_DIR.exists():
        raise FileNotFoundError(f"Upload test tiles to {TILES_DIR}")

    meta_path, meta = find_metadata_file()
    crs = meta["crs"]
    tiles_meta = meta["tiles"]
    px_size_cfg = meta.get("pixel_size_m", [1.0, 1.0])
    px_x = float(px_size_cfg[0])
    px_y = float(px_size_cfg[1])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    encoder_name = ckpt.get("encoder_name", "mit_b2")
    img_size = int(ckpt.get("img_size", meta.get("tile_size_px", 640)))
    num_classes = len(class_names)

    model = smp.Segformer(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    infer_tf = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    valid_tiles = []
    skipped = []
    for path in sorted(TILES_DIR.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        if path.name in tiles_meta:
            valid_tiles.append((path, tiles_meta[path.name]))
        else:
            skipped.append(path.name)

    if not valid_tiles:
        raise FileNotFoundError(f"No matching tiles found in {TILES_DIR} for {meta_path.name}")

    min_x = min(float(info["bounds"][0]) for _, info in valid_tiles)
    min_y = min(float(info["bounds"][1]) for _, info in valid_tiles)
    max_x = max(float(info["bounds"][2]) for _, info in valid_tiles)
    max_y = max(float(info["bounds"][3]) for _, info in valid_tiles)

    global_w = int(round((max_x - min_x) / px_x))
    global_h = int(round((max_y - min_y) / px_y))
    global_transform = from_origin(min_x, max_y, px_x, px_y)

    est_ram_gb = ((num_classes + 1) * global_h * global_w * 4) / (1024 ** 3)
    print(f"Metadata: {meta_path.name} | CRS: {crs} | Tiles: {len(valid_tiles)}")
    print(f"Mosaic size: {global_w}x{global_h} px (~{est_ram_gb:.2f} GB RAM)")
    if skipped:
        print(f"Skipped tiles not in JSON: {len(skipped)}")
    if est_ram_gb > 6.0:
        print("Warning: Large mosaic extent may exceed free Colab RAM. Consider splitting into chunks.")

    prob_sum = np.zeros((num_classes, global_h, global_w), dtype=np.float32)
    weight_map = np.zeros((global_h, global_w), dtype=np.float32)
    window_cache = {}

    for img_path, info in valid_tiles:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        probs = predict_tile_probs(model, rgb, infer_tf, device, use_amp, USE_TTA)
        tile_h, tile_w = probs.shape[1], probs.shape[2]

        win = window_cache.get((tile_h, tile_w))
        if win is None:
            win = make_blend_window(tile_h, tile_w)
            window_cache[(tile_h, tile_w)] = win

        c_x = float(info["transform"][2])
        f_y = float(info["transform"][5])
        col_off = int(round((c_x - min_x) / px_x))
        row_off = int(round((max_y - f_y) / px_y))

        dst_y0 = max(0, row_off)
        dst_x0 = max(0, col_off)
        dst_y1 = min(global_h, row_off + tile_h)
        dst_x1 = min(global_w, col_off + tile_w)

        src_y0 = dst_y0 - row_off
        src_x0 = dst_x0 - col_off
        src_y1 = src_y0 + (dst_y1 - dst_y0)
        src_x1 = src_x0 + (dst_x1 - dst_x0)

        if dst_y1 <= dst_y0 or dst_x1 <= dst_x0:
            continue

        w_slice = win[src_y0:src_y1, src_x0:src_x1]
        prob_sum[:, dst_y0:dst_y1, dst_x0:dst_x1] += probs[:, src_y0:src_y1, src_x0:src_x1] * w_slice
        weight_map[dst_y0:dst_y1, dst_x0:dst_x1] += w_slice

    valid_mask = weight_map > 0
    pred_mask = np.zeros((global_h, global_w), dtype=np.uint8)
    pred_mask[valid_mask] = (prob_sum[:, valid_mask] / weight_map[valid_mask]).argmax(axis=0).astype(np.uint8)

    del prob_sum
    del weight_map

    save_preview(pred_mask, valid_mask, PREVIEW_PATH)

    records = []
    for geom_dict, val in shapes(pred_mask, mask=(pred_mask > 0), connectivity=4, transform=global_transform):
        cls_id = int(val)
        geom = make_valid(shape(geom_dict))
        if SIMPLIFY_TOLERANCE_M > 0:
            geom = make_valid(geom.simplify(SIMPLIFY_TOLERANCE_M, preserve_topology=True))

        for poly in extract_polygons(geom):
            if poly.area < MIN_AREA_M2:
                continue
            records.append({
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "area_m2": round(float(poly.area), 2),
                "geometry": poly,
            })

    if not records:
        print("No polygons detected.")
        return

    if SHP_DIR.exists():
        shutil.rmtree(SHP_DIR)
    SHP_DIR.mkdir(parents=True, exist_ok=True)

    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(SHP_DIR / "predictions_segformer.shp", driver="ESRI Shapefile", encoding="utf-8")
    zip_path = shutil.make_archive(str(ZIP_BASE), "zip", SHP_DIR)

    print("Total polygons:", len(gdf))
    print(gdf.groupby("class_name")["area_m2"].agg(["count", "sum"]).to_string())
    print("Preview:", PREVIEW_PATH)
    print("Archive:", zip_path)
    download_in_colab(zip_path)


if __name__ == "__main__":
    main()
