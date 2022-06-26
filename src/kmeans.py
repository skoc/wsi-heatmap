import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re
import sys
import argparse


def eprint(args):
    sys.stderr.write(str(args) + "\n")


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
