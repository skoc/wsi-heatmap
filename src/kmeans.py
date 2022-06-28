import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import sys
import argparse
from PIL import Image
import random

def eprint(args):
    sys.stderr.write(str(args) + "\n")

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_features", required=True, help="Path to extracted features (to folder)")
    parser.add_argument("--path_grid_tiles", required=True, help="Path to grid tiles of WSIs (to folder)")
    parser.add_argument("--path_write", required=False, help="Write path of kmeans csv", default="data/kmeans/")
    args = parser.parse_args()

    path_features = args.path_features
    path_grid_tiles = args.path_grid_tiles
    path_write = args.path_write

    lst_clusters = [3, 5, 8]
    random_state = 2022

    eprint(f"Using these feature metadata files:")
    for f in os.listdir(path_features):
        if f.startswith('features_metadata_'):
            eprint(f)

    dfs = [pd.read_csv(os.path.join(path_features, f)) for f in os.listdir(path_features) if
           f.startswith('features_metadata_')]

    df_multiple_wsi = pd.concat(dfs, axis=0, join='inner').reset_index(drop=True)  # .sort_index()
    eprint(f"\nShape: {df_multiple_wsi.shape}")
    df_multiple_wsi.head(2)

    df_multiple_wsi['path'] = df_multiple_wsi.apply(lambda row: path_grid_tiles + str(row.wsi) + "/" + row.fname,
                                                    axis=1)

    df_multiple_wsi.rename(columns={'wsi': 'id'}, inplace=True)
    df_multiple_wsi['label'] = 0
    eprint(f"\nShape: {df_multiple_wsi.shape}")

    feature_imagenet = df_multiple_wsi.iloc[:, :-4]
    scaled_results_tg = StandardScaler().fit_transform(feature_imagenet)

    for n_cluster in lst_clusters:
        df_multiple_wsi[f'kmeans_labels_{n_cluster}'] = KMeans(n_clusters=n_cluster, random_state=random_state).fit(
            scaled_results_tg).labels_

    regx = re.compile('\_[0-9,\-]*\.')
    df_multiple_wsi['coord'] = df_multiple_wsi.path.apply(lambda x: regx.findall(x)[-1][1:-1])
    df_multiple_wsi['wsi'] = df_multiple_wsi['id'].astype(str)
    df_multiple_wsi['x'] = df_multiple_wsi.coord.apply(lambda x: x.split('-')[0]).astype(int)
    df_multiple_wsi['y'] = df_multiple_wsi.coord.apply(lambda x: x.split('-')[1]).astype(int)
    df_multiple_wsi['x2'] = df_multiple_wsi.coord.apply(lambda x: x.split('-')[2]).astype(int)
    df_multiple_wsi['y2'] = df_multiple_wsi.coord.apply(lambda x: x.split('-')[3]).astype(int)

    df_multiple_wsi['id'] = df_multiple_wsi['id'].apply(str)
    df_multiple_wsi['wsi'] = df_multiple_wsi['wsi'].apply(str)

    path_save = os.path.join(path_write, "output_cluster_mix_kmeans.csv")
    df_multiple_wsi.to_csv(path_save, index=False)

    eprint(f"Kmeans clustering score attached metadata saved: {path_save}!")

    # Random Tile Plots
    lst_wsi = df_multiple_wsi.wsi.tolist()
    path_write_random_grid = "data/kmeans/visualizations_grid/"

    eprint(f"Random Tile Plots...")

    for id_img in lst_wsi:
        for n_cluster in [3,5,8]:
            tile_count = 10
            size_img = 500
            expand = int(tile_count / (size_img / 100) * n_cluster * 100)  # + (n_cluster - 1)*5

            gb = df_multiple_wsi[df_multiple_wsi.id == id_img][['id', 'path', f'kmeans_labels_{n_cluster}']].reset_index(drop=True).groupby(
                [f'kmeans_labels_{n_cluster}'])
            ls = []

            for _ in range(tile_count):
                for index, frame in gb:
                    ls.append(frame[frame['path'] == random.choice(frame['path'].unique())].sample(n=1))

            df_img_print = pd.concat(ls)
            df_img_print = df_img_print.sort_values(by=f'kmeans_labels_{n_cluster}', ascending=True).reset_index(drop=True)

            new_im = Image.new('RGB', (size_img, expand))
            m, k, load = 0, 0, 0

            for i, row in df_img_print.iterrows():
                im = Image.open(row.path)
                im.thumbnail((100, 100))
                # paste the image at location i,j:
                if (i != 0) and (i % 5 == 0):
                    k += 1
                    m = 0
                    if (i % 10 == 0):
                        load = 5

                y_axis = k * 100

                if load > 0:
                    y_axis += 5
                    new_im.paste(im, (m * 100, y_axis))
                    load -= 1
                else:
                    new_im.paste(im, (m * 100, y_axis))
                m += 1
            path_write_random_grid = os.path.join(path_write_random_grid, f"{id_img}")
            new_im.save(os.path.join(mkdir_if_not_exist(path_write_random_grid), f"grid_tile_{id_img}_kmeans-{n_cluster}.jpg"))
            eprint(f"Saved: {path_write_random_grid}, grid_tile_{id_img}_kmeans-{n_cluster}.jpg")

