import json
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

INPUT_DIR = Path("Ready")
OUTPUT_DIR = Path("Ready_for_learning")
GEOREF_FILE = Path("Ready_for_learning.json")

TILE_SIZE = 640
OVERLAP = 128
MIN_VALID_RATIO = 0.25
NODATA_VALUES = (0, 255)
IMG_FORMAT = "jpg"


def get_offsets(size, tile_size, overlap):
    if size <= tile_size:
        return [0]
    step = tile_size - overlap
    offsets = list(range(0, size - tile_size, step))
    last = size - tile_size
    if not offsets or offsets[-1] != last:
        offsets.append(last)
    return offsets


def save_image(tile, path):
    rgb = np.transpose(tile[:3], (1, 2, 0))
    img = Image.fromarray(rgb, mode="RGB")
    if IMG_FORMAT == "jpg":
        img.save(path, format="JPEG", quality=95, subsampling=0)
    else:
        img.save(path, format="PNG")


def main():
    tif_files = sorted(INPUT_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    GEOREF_FILE.parent.mkdir(parents=True, exist_ok=True)

    georef = {
        "crs": None,
        "pixel_size_m": None,
        "pixel_area_m2": None,
        "tile_size_px": TILE_SIZE,
        "overlap_px": OVERLAP,
        "tiles": {},
    }

    total_saved = 0
    total_skipped = 0

    for path in tif_files:
        tile_dir = OUTPUT_DIR / path.stem
        tile_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        skipped = 0

        with rasterio.open(path) as src:
            if georef["crs"] is None:
                px_w, px_h = src.res
                georef["crs"] = src.crs.to_string()
                georef["pixel_size_m"] = [round(px_w, 6), round(px_h, 6)]
                georef["pixel_area_m2"] = round(px_w * px_h, 6)

            rows = get_offsets(src.height, TILE_SIZE, OVERLAP)
            cols = get_offsets(src.width, TILE_SIZE, OVERLAP)

            for r in rows:
                for c in cols:
                    win = Window(c, r, TILE_SIZE, TILE_SIZE)

                    masks = src.read_masks(window=win, boundless=True)
                    valid = np.all(masks > 0, axis=0)
                    if valid.mean() < MIN_VALID_RATIO:
                        skipped += 1
                        continue

                    tile = src.read(window=win, boundless=True, fill_value=0)
                    for val in NODATA_VALUES:
                        valid &= ~np.all(tile == val, axis=0)

                    if valid.mean() < MIN_VALID_RATIO:
                        skipped += 1
                        continue

                    name = f"{path.stem}_y{r:06d}_x{c:06d}.{IMG_FORMAT}"
                    save_image(tile, tile_dir / name)

                    georef["tiles"][name] = {
                        "source": path.name,
                        "row_off": r,
                        "col_off": c,
                        "transform": list(src.window_transform(win))[:6],
                        "bounds": list(src.window_bounds(win)),
                    }
                    saved += 1

        total_saved += saved
        total_skipped += skipped
        print(f"{path.stem}: saved {saved}, skipped {skipped}")

    GEOREF_FILE.write_text(json.dumps(georef, indent=2), encoding="utf-8")
    print(f"Done. Total saved: {total_saved}, skipped: {total_skipped}")


if __name__ == "__main__":
    main()