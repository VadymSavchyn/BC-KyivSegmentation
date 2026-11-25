import math
from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

INPUT_DIR = Path("Clip_Out")
OUTPUT_DIR = Path("Ready")
TARGET_RES = 1.0


def get_common_grid(files, res):
    bounds = []
    crs_values = set()

    for path in files:
        with rasterio.open(path) as src:
            if not src.crs:
                raise ValueError(f"Missing CRS in {path.name}")
            bounds.append(src.bounds)
            crs_values.add(src.crs.to_string())

    if len(crs_values) != 1:
        raise ValueError(f"Mismatched CRS: {crs_values}")

    crs = crs_values.pop()
    west = math.ceil(max(b.left for b in bounds))
    south = math.ceil(max(b.bottom for b in bounds))
    east = math.floor(min(b.right for b in bounds))
    north = math.floor(min(b.top for b in bounds))

    width = round((east - west) / res)
    height = round((north - south) / res)
    transform = from_origin(west, north, res, res)

    return crs, transform, width, height


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tif_files = sorted(INPUT_DIR.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No .tif files in {INPUT_DIR}")

    crs, transform, width, height = get_common_grid(tif_files, TARGET_RES)
    print(f"Grid: {width}x{height} ({crs}, {TARGET_RES}m)")

    profile = {
        "driver": "GTiff",
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
        "count": 3,
        "dtype": "uint8",
        "compress": "deflate",
        "tiled": True,
    }

    for path in tif_files:
        out_path = OUTPUT_DIR / f"{path.stem}_{TARGET_RES}m.tif"

        with rasterio.open(path) as src:
            with WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
            ) as vrt:
                data = vrt.read()

        rgb = np.repeat(data, 3, axis=0) if data.shape[0] == 1 else data[:3]

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(rgb)

        print(f"Saved {out_path.name}")


if __name__ == "__main__":
    main()