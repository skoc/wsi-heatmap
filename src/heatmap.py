from pathlib import Path
from typing import (Dict, List, Tuple)
import openslide
from PIL import Image
import math
import numpy as np
from imageio import (imread, imsave)
from datetime import datetime
import pandas as pd
import argparse
import sys
import os


def eprint(args):
    sys.stderr.write(str(args) + "\n")


def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir


def color_to_np_color(color: str) -> np.ndarray:
    """
    Convert strings to NumPy colors.
    Args:
        color: The desired color as a string.
    Returns:
        The NumPy ndarray representation of the color.
    """
    colors = {
        "white": np.array([255, 255, 255]),
        "pink": np.array([255, 108, 180]),
        "black": np.array([0, 0, 0]),
        "red": np.array([255, 0, 0]),
        "purple": np.array([225, 225, 0]),
        "yellow": np.array([255, 255, 0]),
        "orange": np.array([255, 127, 80]),
        "blue": np.array([0, 0, 255]),
        "green": np.array([0, 255, 0])
    }
    return colors[color]


def add_predictions_to_image(
        xy_to_pred_class: Dict[Tuple[str, str], Tuple[str, float]],
        image: np.ndarray, prediction_to_color: Dict[str, np.ndarray],
        patch_size: int) -> np.ndarray:
    """
    Overlay the predicted dots (classes) on the WSI.

    Args:
        xy_to_pred_class: Dictionary mapping coordinates to predicted class along with the confidence.
        image: WSI to add predicted dots to.
        prediction_to_color: Dictionary mapping string color to NumPy ndarray color.
        patch_size: Size of the patches extracted from the WSI.

    Returns:
        The WSI with the predicted class dots overlaid.
    """
    counter = 0
    for x, y in xy_to_pred_class.keys():
        prediction, __ = xy_to_pred_class[x, y]
        x = int(x)
        y = int(y)

        # Enlarge the dots so they are visible at larger scale.
        start = round((0.6 * patch_size) / 2)
        end = round((1.3 * patch_size) / 2)
        image[y + start:y + end, x + start:x + end, :] = prediction_to_color[str(prediction)]
        counter += 1
    print(counter)
    return image


def get_xy_to_pred_class(window_prediction_folder: Path,
                         img_name: str, scale_factor: int, wsi_id: str,
                         target_label: str
                         ) -> Dict[Tuple[str, str], Tuple[str, float]]:
    """
    Find the dictionary of predictions.

    Args:
        window_prediction_folder: Path to the folder containing a CSV file with the predicted classes.
        img_name: Name of the image to find the predicted classes for.

    Returns:
        A dictionary mapping image coordinates to the predicted class and the confidence of the prediction.
    """
    xy_to_pred_class = {}
    df = pd.read_csv(window_prediction_folder, low_memory=False)
    df = df[df.wsi == wsi_id].reset_index(drop=True)

    for index, row in df.iterrows():
        x = int(row['x']) // scale_factor
        y = int(row['y']) // scale_factor
        pred_class = row[target_label]
        confidence = row[target_label]

        xy_to_pred_class[(x, y)] = (pred_class, confidence)

    return xy_to_pred_class


def get_image_paths(folder: Path) -> List[Path]:
    """
    Find the full paths of the images in a folder.

    Args:
        folder: Folder containing images (assume folder only contains images).

    Returns:
        A list of the full paths to the images in the folder.
    """
    IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".svs", ".tif", ".tiff"]
    return sorted([folder.joinpath(f.name) for f in folder.iterdir() if
                   ((folder.joinpath(f.name).is_file()) and (".DS_Store" not in f.name) and (
                               f.suffix.casefold() in IMAGE_EXTS))], key=str)


def get_subfolder_paths(folder: Path) -> List[Path]:
    """
    Find the paths of subfolders.

    Args:
        folder: Folder to look for subfolders in.

    Returns:
        A list containing the paths of the subfolders.
    """
    return sorted([folder.joinpath(f.name) for f in folder.iterdir() if
                   ((folder.joinpath(f.name).is_dir()) and (".DS_Store" not in f.name))], key=str)


def get_all_image_paths(master_folder: Path) -> List[Path]:
    """
    Finds all image paths in subfolders.

    Args:
        master_folder: Root folder containing subfolders.

    Returns:
        A list of the paths to the images found in the folder.
    """
    all_paths = []
    subfolders = get_subfolder_paths(folder=master_folder)
    if len(subfolders) > 1:
        for subfolder in subfolders:
            all_paths += get_image_paths(folder=subfolder)
    else:
        all_paths = get_image_paths(folder=master_folder)
    return all_paths


def visualize(whole_slide: Path,
              preds_folder: Path,
              vis_folder: Path,
              classes: List[str], num_classes: int, colors: Tuple[str],
              patch_size: int,
              scale_factor: int, wsi_id: str,
              target_label="kmeans_labels_3") -> None:
    """
    Main function for visualization.

    Args:
        whole_slide: Path to WSI.
        preds_folder: Path containing the predicted classes.
        vis_folder: Path to output the WSI with overlaid classes to.
        classes: Names of the classes in the dataset.
        num_classes: Number of classes in the dataset.
        colors: Colors to use for visualization.
        patch_size: Size of the patches extracted from the WSI.
    """
    # Find list of WSI.
    #     whole_slides = get_all_image_paths(master_folder=wsi_folder)
    #     print(f"{len(whole_slides)} whole slides found from {wsi_folder}")
    #     prediction_to_color = {
    #         classes[i]: color_to_np_color(color=colors[i])
    #         for i in range(num_classes)
    #     }
    prediction_to_color = {"0": [0, 0, 255],
                           "1": [255, 0, 0],
                           "2": [0, 0, 0],
                           "3": [255, 255, 255],
                           "4": [0, 255, 0],
                           "5": [255, 127, 80],
                           "6": [255, 108, 180],
                           "7": [225, 225, 0]}
    #                              "-1": [255, 255,   0]}
    # Go over all of the WSI.
    #     for whole_slide in whole_slides:
    # Read in the image.
    print(f"WSI: {whole_slide}")

    whole_slide_numpy = imread(uri=whole_slide)[..., [0, 1, 2]]

    print(f"visualizing {whole_slide} "
          f"of shape {whole_slide_numpy.shape}")

    assert whole_slide_numpy.shape[
               2] == 3, f"Expected 3 channels while your image has {whole_slide_numpy.shape[2]} channels."

    # Save it.
    date = datetime.now().strftime("%Y%m%d-%I%M%S")
    output_path = Path(
        f"{vis_folder.joinpath(whole_slide.name).with_suffix('')}"
        f"_predictions_{date}.jpg")

    # Confirm the output directory exists.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    imsave(uri=output_path,
           im=add_predictions_to_image(
               xy_to_pred_class=get_xy_to_pred_class(
                   window_prediction_folder=preds_folder,
                   img_name=whole_slide.name,
                   scale_factor=scale_factor,
                   wsi_id=wsi_id,
                   target_label=target_label
               ),
               image=whole_slide_numpy,
               prediction_to_color=prediction_to_color,
               patch_size=patch_size))

    print(f"find the visualizations in {vis_folder}")


def slide_to_scaled_pil_image(path_wsi, path_write='data/wsi-scaled/', scale_factor=32):
    slide = openslide.OpenSlide(path_wsi)
    fname = path_wsi.split('/')[-1].split('.')[0]

    output_folder = mkdir_if_not_exist(path_write)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / scale_factor)
    new_h = math.floor(large_h / scale_factor)
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")

    path_scaled = output_folder + fname + '.png'
    whole_slide_image.resize((new_w, new_h), Image.BILINEAR).save(path_scaled)
    eprint(f"Saved scaled img here: {path_scaled}")

    return path_scaled


if __name__ == '__main__':

    lst_clusters = [3, 5, 8]
    lst_colors = ('blue', 'red', "black", "purple", "green", "orange", "pink", "white", "yellow")

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_wsi", required=True, help="WSI")
    parser.add_argument("--path_cluster_metadata", required=True, help="Path of CSV with Scores")
    parser.add_argument("--tile_size", required=False, help="Tile size", type=int, default=1024)
    parser.add_argument("--scale_factor", required=False, help="Tile size", type=int, default=32)
    parser.add_argument("--scale_wsi", required=False, help="Scale WSI with given scale factor", type=bool,
                        default=True)
    parser.add_argument("--path_write", required=False, help="Path of extracted features", default="data/heatmap")
    args = parser.parse_args()

    path_wsi = args.path_wsi
    path_cluster_metadata = args.path_cluster_metadata
    tile_size = args.tile_size
    scale_wsi = args.scale_wsi
    scale_factor = args.scale_factor
    path_write = args.path_write
    patch_size = tile_size // scale_factor

    fname_wsi = path_wsi.split("/")[-1].split(".")[0]
    path_wsi_scaled = None

    if scale_wsi:
        path_wsi_scaled = Path(
            slide_to_scaled_pil_image(path_wsi, path_write='data/wsi-scaled/', scale_factor=scale_factor))
    else:
        path_wsi_scaled = Path(os.path.join('data/wsi-scaled/', fname_wsi + '.png'))

    df_multiple = pd.read_csv(path_cluster_metadata)

    for n_clusters in lst_clusters:
        lst_label_idx = df_multiple[f'kmeans_labels_{n_clusters}'].value_counts().index.tolist()

        visualize(path_wsi_scaled,
                  path_cluster_metadata,
                  Path(path_write),
                  lst_label_idx,
                  n_clusters,
                  lst_colors,
                  patch_size=patch_size,
                  scale_factor=scale_factor,
                  wsi_id=fname_wsi,
                  target_label=f"kmeans_labels_{n_clusters}")
