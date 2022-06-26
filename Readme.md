# WSI Clustering Heatmap with ResNet Features 


## Setup

```
git clone https://github.com/skoc/wsi-heatmap.git
cd wsi-heatmap
conda env create -f environment.yml
``` 

### Dataset

```bash
wsi-heatmap 
└── data 
      └── wsi 
            ├── <WSI_ID>.tiff
            └── <WSI_ID>.svs
```

### Grid Tiling

To prepare tiles of a WSI:
```
python src/tiling.py --path_wsi data/wsi/<WSI> --level=1 
--tile_size=512 --tissue_percent=90 --check_tissue=True 
```
```
optional arguments:
  -h, --help            show this help message and exit
  --path_wsi PATH_WSI   Path of a WSI to be tiled
  --path_output PATH_OUTPUT
                        Path of output tiled images
  --tile_size TILE_SIZE
                        Tile size
  --level LEVEL         WSI extraction level (0 is the highest resolution)
  --check_tissue CHECK_TISSUE
                        Check tiles in terms of tissue density
  --tissue_percent TISSUE_PERCENT
                        Min percentage of tissue is required to save the tiles
  --pixel_overlap PIXEL_OVERLAP
                        Pixel overlap while tiling
  --prefix PREFIX       Prefix to use while saving tiles
  --suffix SUFFIX       Extension of tiles
  --path_metadata PATH_METADATA
                        Path of output tiled images
```

To find the output of the grid tiling:
```bash
wsi-heatmap 
└── data 
      └── grid_tiles
            ├── <WSI_ID>/
```


### Feature Extraction

To extract ResNet features of WSI tiles:
```
python src/feature_extraction.py --path_metadata data/metadata/metadata_<WSI>.csv
```
```
optional arguments:
  -h, --help            show this help message and exit
  --path_model PATH_MODEL
                        Path of a trained weights if there is any
                        (default:ImageNet)
  --path_metadata PATH_METADATA
                        Path of WSI tiles metadata
  --batch_size BATCH_SIZE
                        Batch size
  --path_write PATH_WRITE
                        Path of extracted features


```

To find the output of the feature extraction:
```bash
wsi-heatmap 
└── data 
      └── features
            ├── features_metadata_<WSI>.csv
```

### Clustering

To cluster extracted tile features:
```
python src/kmeans.py --path_features data/features/ --path_grid_tiles data/grid_tiles/  
```
```
optional arguments:
  -h, --help            show this help message and exit
  --path_features PATH_FEATURES
                        Path to extracted features (to folder)
  --path_grid_tiles PATH_GRID_TILES
                        Path to grid tiles of WSIs (to folder)
  --path_write PATH_WRITE
                        Write path of kmeans csv
```

To find the output of the clustering:
```bash
wsi-heatmap 
└── data 
      └── kmeans
            ├── output_cluster_mix_kmeans.csv
```

### Heatmap

To visualize heatmap of clusters on a scaled WSI:
```
python src/heatmap.py --path_wsi data/wsi/<WSI> --path_cluster_metadata data/kmeans/output_cluster_mix_kmeans.csv 
```
```
optional arguments:
  -h, --help            show this help message and exit
  --path_wsi PATH_WSI   WSI
  --path_cluster_metadata PATH_CLUSTER_METADATA
                        Path of CSV with Scores
  --tile_size TILE_SIZE
                        Tile size
  --scale_factor SCALE_FACTOR
                        Tile size
  --scale_wsi SCALE_WSI
                        Scale WSI with given scale factor
  --path_write PATH_WRITE
                        Path of extracted features
```

To find the output of the heatmap:
```bash
wsi-heatmap 
└── data 
      └── heatmap
            ├── <WSI_ID>_predictions_<date>.jpg
```

## License


## Reference

