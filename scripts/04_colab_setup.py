import os
import subprocess
import sys
from pathlib import Path

PACKAGES = [
    "segmentation-models-pytorch",
    "albumentations",
    "roboflow",
    "geopandas",
    "rasterio",
    "shapely",
    "pyyaml",
]

WORK_DIRS = [
    Path("/content/test_tiles"),
    Path("/content/previews"),
    Path("/content/outputs"),
]


def install_packages(packages):
    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages]
    subprocess.check_call(cmd)


def prepare_folders(folders):
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def get_ram_gb():
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round((pages * page_size) / (1024 ** 3), 1)
    except (AttributeError, ValueError, OSError):
        return None


def main():
    install_packages(PACKAGES)
    prepare_folders(WORK_DIRS)

    import torch

    has_cuda = torch.cuda.is_available()
    device_label = torch.cuda.get_device_name(0) if has_cuda else "CPU"
    ram_gb = get_ram_gb()

    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device_label}")
    if ram_gb is not None:
        print(f"RAM: {ram_gb} GB")
    print("Folders ready:", ", ".join(str(p) for p in WORK_DIRS))


if __name__ == "__main__":
    main()
