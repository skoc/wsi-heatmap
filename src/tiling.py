import argparse
import time
import logging
import os
import sys
from glob import glob
import pandas as pd
from histolab.slide import Slide
from histolab.tiler import GridTiler

logging.basicConfig(filename='logs/tiling.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

def eprint(args):
    sys.stderr.write(str(args) + "\n")

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir

def get_grid_tiler(path_wsi,
                   path_output="data/grid_tiles",
                   tile_size=512,
                   level=0,
                   check_tissue=True,
                   tissue_percent=50,
                   pixel_overlap=False,
                   prefix="grid",
                   suffix=".png"):
    start_time = time.time()
    fname_wsi = path_wsi.split("/")[-1].split(".")[0]
    processed_path = mkdir_if_not_exist(os.path.join(path_output, fname_wsi))

    # Tiling
    grid_tiles_extractor = GridTiler(
        tile_size=(tile_size, tile_size),
        level=level,  # 0.25mpp is 40x, level=1 takes 20x
        check_tissue=check_tissue,
        tissue_percent=tissue_percent,
        pixel_overlap=pixel_overlap,  # default
        prefix=prefix,  # save tiles in the "grid" subdirectory of slide's processed_path
        suffix=suffix  # default,
        #    mpp = 0.5
    )

    slide_wsi = Slide(path_wsi, processed_path=processed_path)

    logging.info(f"Slide name: {slide_wsi.name}")
    logging.info(f"Dimensions: {slide_wsi.dimensions}")

    logging.info(f"Grid Tiles Extraction of {fname_wsi} is started...")
    grid_tiles_extractor.extract(slide_wsi)
    logging.info(f" Grid Tiles Extraction of {fname_wsi} is done!")

    end_time = time.time() - start_time
    logging.info(f"Grid Tiles of WSI:{fname_wsi} is saved in {processed_path}! (Time it takes {round(end_time, 2)})")

    return processed_path

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_wsi", required=False, help="Path of a WSI to be tiled", type=str)
    parser.add_argument("--path_output", required=False, help="Path of output tiled images", default="data/grid_tiles")
    parser.add_argument("--tile_size", required=False, help="Tile size", type=int, default=512) # (level0, 512 -> 512x512) (level1, 512 -> 1024x1024)
    parser.add_argument("--level", required=False, help="WSI extraction level (0 is the highest resolution)", type=int, default=0)
    parser.add_argument("--check_tissue", required=False, help="Check tiles in terms of tissue density", type=bool, default=True)
    parser.add_argument("--tissue_percent", required=False, help="Min percentage of tissue is required to save the tiles", type=int, default=50)
    parser.add_argument("--pixel_overlap", required=False, help="Pixel overlap while tiling", type=int, default=0)
    parser.add_argument("--prefix", required=False, help="Prefix to use while saving tiles", type=str, default="grid")
    parser.add_argument("--suffix", required=False, help="Extension of tiles", type=str, default=".png")
    parser.add_argument("--path_metadata", required=False, help="Path of output tiled images", default="data/metadata")

    args = parser.parse_args()

    path_wsi = args.path_wsi
    path_output = args.path_output
    tile_size = args.tile_size
    level = args.level
    check_tissue = args.check_tissue
    tissue_percent = args.tissue_percent
    pixel_overlap = args.pixel_overlap
    prefix = args.prefix
    suffix = args.suffix
    path_metadata = args.path_metadata

    processed_path = get_grid_tiler(path_wsi,
                                    path_output,
                                    tile_size,
                                    level,
                                    check_tissue,
                                    tissue_percent,
                                    pixel_overlap,
                                    prefix,
                                    suffix)

    # Get List of Files with Specific Extension - tiles
    lst_tiles = [f for f in glob(processed_path + f"*/*{suffix}")]
    df_tiles = pd.DataFrame({"path": lst_tiles})
    eprint(f"Shape: {df_tiles.shape}")

    df_tiles['tile'] = df_tiles['path'].apply(lambda x: x.split('/')[-1])
    df_tiles['wsi'] = df_tiles['path'].apply(lambda x: x.split('/')[-2].split('.')[0])
    df_tiles['x'] = df_tiles['tile'].apply(lambda x: int(x.split('_')[-1].split('.')[0].split('-')[0]))
    df_tiles['y'] = df_tiles['tile'].apply(lambda x: int(x.split('_')[-1].split('.')[0].split('-')[1]))
    df_tiles['label'] = 'dummy'  # dummy label for selfsupervised feature extraction

    write_metadata = os.path.join(path_metadata, "metadata_"+path_wsi.split('/')[-1]+".csv")

    df_tiles.to_csv(write_metadata, index=False)

    eprint(f"Shape after extension: {df_tiles.shape}")
    df_tiles.head()


