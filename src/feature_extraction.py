import torchvision
import torch
import os
import argparse
import pandas as pd
import sys

sys.path.insert(0,'dataloader.py')

import torchvision
from torch.utils.data import DataLoader
from dataloader import TileDataset

def eprint(args):
    sys.stderr.write(str(args) + "\n")


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}

    assert weights, 'No weight could be loaded...'

    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model


def get_pretrained(path_model=None,
                   return_preactivation=True,
                   num_classes=3,
                   device='cpu'):

    if path_model:
        model = torchvision.models.__dict__['resnet18'](pretrained=False)
        state = torch.load(path_model, map_location='cuda:0')
        state_dict = state['model_state_dict'] if 'model_state_dict' in state.keys() else state['state_dict']

        for key in list(state_dict.keys()):
            state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

        model = load_model_weights(model, state_dict)
        eprint(f"[INFO] Loaded model with given model: {path_model} !")

    elif path_model is None:

        model = torchvision.models.__dict__['resnet18'](pretrained=True)
        eprint(f"[INFO] Loaded model with ImageNet!")

    else:

        eprint("[INFO] Undefined!")

    model = model.to(device)

    if return_preactivation:
        model.fc = torch.nn.Sequential()
    else:
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    return model

if __name__ == '__main__':

    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_model", required=False, help="Path of a trained weights if there is any (default:ImageNet)", default=None)
    parser.add_argument("--path_metadata", required=True, help="Path of WSI tiles metadata")
    parser.add_argument("--batch_size", required=False, help="Batch size", type=int, default=32)
    parser.add_argument("--path_write", required=False, help="Path of extracted features", default="data/features")
    args = parser.parse_args()

    path_model = args.path_model #'models/tenpercent_resnet18.ckpt'
    path_metadata = args.path_metadata
    batch_size = args.batch_size
    path_write = args.path_write

    # GPU or CPU
    USE_GPU = True
    dtype = torch.float32

    # GPU or CPU
    device = torch.device('cuda') if (USE_GPU and torch.cuda.is_available()) else torch.device('cpu')
    eprint(f'[INFO] using device: {device}')

    # Load Model
    model = get_pretrained(path_model=path_model, device=device)

    # Dataloader
    dataset = TileDataset(metadata=path_metadata)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    eprint("[INFO] Dataloader is prepared!")

    # Feature Extraction
    df_features = pd.DataFrame()

    for data_dict in dataloader:
        # Transferring tensor to GPU
        x = data_dict['image'].to(device)
        output = model(x)
        output_pd = output.to(torch.device('cpu')).detach().numpy()

        df = pd.DataFrame(output_pd)
        df['fname'] = [name.split('/')[-1] for name in data_dict['name_img']]
        df['wsi'] = data_dict['wsi']
        df_features = df_features.append(df)

    eprint("[INFO] Feature Extraction is completed!")

    df_features = df_features.reset_index(drop=True)
    eprint(f"[INFO] Shape df_features: {df_features.shape}")

    path_features = os.path.join(path_write, "features_"+path_metadata.split("/")[-1])
    df_features.to_csv(path_features, index=False)
    eprint(f"[INFO] Extracted features are writen at: {path_features}!")