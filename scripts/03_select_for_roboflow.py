import random
import re
import shutil
from pathlib import Path

TILES_DIR = Path("Ready_for_learning")
OUTPUT_DIR = Path("Roboflow_Upload")
SEED = 42

TRAIN_MAX_Y = 3584
VALID_MIN_Y = 4608

TRAIN_COUNTS = {
    "1941": 25,
    "1984": 25,
    "2005": 50,
}

VALID_COUNTS = {
    "1941": 8,
    "1984": 8,
    "2005": 12,
}

TEST_YEAR = "2025"
TEST_UNSEEN_AREA_COUNT = 10
TEST_GENERAL_COUNT = 10


def parse_y(path):
    match = re.search(r"_y(\d+)_x\d+", path.stem)
    if not match:
        raise ValueError(f"Cannot parse y coordinate from {path.name}")
    return int(match.group(1))


def find_year_dir(year):
    matches = [p for p in TILES_DIR.iterdir() if p.is_dir() and p.name.startswith(year)]
    if not matches:
        raise FileNotFoundError(f"Folder for {year} not found in {TILES_DIR}")
    return sorted(matches)[0]


def copy_sample(files, count, dest_dir, rng):
    dest_dir.mkdir(parents=True, exist_ok=True)
    chosen = rng.sample(files, min(count, len(files)))
    for src in chosen:
        shutil.copy2(src, dest_dir / src.name)
    return len(chosen)


def main():
    rng = random.Random(SEED)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    train_dir = OUTPUT_DIR / "train"
    valid_dir = OUTPUT_DIR / "valid"
    test_dir = OUTPUT_DIR / "test"

    for year in TRAIN_COUNTS:
        year_dir = find_year_dir(year)
        images = sorted(year_dir.glob("*.jpg")) + sorted(year_dir.glob("*.png"))

        train_pool = [p for p in images if parse_y(p) <= TRAIN_MAX_Y]
        valid_pool = [p for p in images if parse_y(p) >= VALID_MIN_Y]

        n_train = copy_sample(train_pool, TRAIN_COUNTS[year], train_dir, rng)
        n_valid = copy_sample(valid_pool, VALID_COUNTS[year], valid_dir, rng)
        print(f"{year}: train={n_train}, valid={n_valid}")

    test_year_dir = find_year_dir(TEST_YEAR)
    test_images = sorted(test_year_dir.glob("*.jpg")) + sorted(test_year_dir.glob("*.png"))

    test_unseen = [p for p in test_images if parse_y(p) >= VALID_MIN_Y]
    test_general = [p for p in test_images if parse_y(p) <= TRAIN_MAX_Y]

    n_test_unseen = copy_sample(test_unseen, TEST_UNSEEN_AREA_COUNT, test_dir, rng)
    n_test_general = copy_sample(test_general, TEST_GENERAL_COUNT, test_dir, rng)
    print(f"{TEST_YEAR}: test={n_test_unseen + n_test_general} (unseen_area={n_test_unseen}, main_area={n_test_general})")

    print(f"Done. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()