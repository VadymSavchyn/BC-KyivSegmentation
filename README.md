Kyiv Aerial Imagery Segmentation

Semantic segmentation of multi-temporal aerial and satellite imagery of Kyiv (1941, 1984, 2005, and 2025) using SegFormer (mit_b2) for land-cover mapping and area calculation in ArcGIS Pro.

Project Structure

.
├── scripts/
│   ├── 01_align_rasters.py
│   ├── 02_tile_dataset.py
│   ├── 03_select_for_roboflow.py
│   ├── 04_colab_setup.py
│   ├── 05_prepare_dataset.py
│   ├── 06_train_segformer.py
│   └── 07_infer_to_shapefile.py
├── .gitignore
├── requirements.txt
└── README.md


Workflow

Source data: 4 time periods (1941, 1984, 2005, 2025), 1 m/px resolution, EPSG:32636, tiled into 640x640 px chunks with 128 px overlap.

1. Local preprocessing

Install dependencies:

pip install -r requirements.txt


Run scripts in order:

01_align_rasters.py: aligns source GeoTIFF rasters across time periods.

02_tile_dataset.py: cuts imagery into overlapping tiles and creates metadata.json.

03_select_for_roboflow.py: selects tiles for labeling in Roboflow.

2. Training and inference (Google Colab)

Run on a GPU runtime (T4):

04_colab_setup.py: installs packages and creates working folders.

05_prepare_dataset.py: downloads the Roboflow dataset and converts annotations to PNG masks.

06_train_segformer.py: trains SegFormer (mit_b2) with CrossEntropy + Dice loss for 60 epochs and saves best_segformer.pth.

07_infer_to_shapefile.py: runs tiled inference with Hann window blending and 4x flip TTA, then exports predictions_segformer.shp as a zip archive.

3. ArcGIS Pro post-processing

Load predictions_segformer.shp into ArcGIS Pro.

Run Simplify Polygon (Douglas-Peucker, 1.0 - 1.5 m tolerance, resolve topological errors).

Optionally run Regularize Building Footprint on buildings.

Generate the other class using Pairwise Erase against the study area boundary, merge layers, and calculate final areas in EPSG:32636.
