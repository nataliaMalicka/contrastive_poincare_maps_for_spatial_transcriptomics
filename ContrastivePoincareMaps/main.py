import os
import argparse

import pandas as pd

from helpers.data import make_dataset
import random
import numpy as np
import torch
from helpers.train import train_model
from helpers.visualize import plotPoincareDisc, plot_poincare_disc
from model.poincare_loss import poincare_root, poincare_translation
import json
import seaborn as sns

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_col_palette(labels):
    labels_order = np.unique(labels).astype(str)
    col_palette = sns.color_palette("hls", len(labels_order)).as_hex()
    col_palette = dict(zip(labels_order, col_palette))
    return col_palette


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", default=None)
    parser.add_argument("--corr_rate", type=float, help="Rate of feature perturbation (range 0 to 1)", default=0.2)
    parser.add_argument("--emb_dim", type=int, help="Embedding dimension", default=128)
    parser.add_argument("--n_enc_layers", type=int, help="Number of encoder layers", default=1)
    parser.add_argument("--n_hyp_layers", type=int, help="Number of hyperbolic layers", default=1)
    parser.add_argument("--lr_proj", type=float, help="Learning rate for projection layers", default=1e-3)
    parser.add_argument("--lr_hyp", type=float, help="Learning rate for hyperbolic layers", default=1e-4)
    parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=1000)
    parser.add_argument("--early_stop", type=float, help="Early stopping criterion", default=1e-3)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=128)
    parser.add_argument("--n_comp_PCA", type=int, help="Number of PCA components; None if disabled", default=None)
    parser.add_argument("--normalise", type=bool, help="Apply z-transform?", default=False)
    parser.add_argument("--data_dir", type=str, help="Path to the data directory", default="Data/")
    parser.add_argument("--fname", type=str, help="Name of the dataset file", default="mouse_gastrulation")
    parser.add_argument("--root", type=str, help= "Root name", default="root")
    parser.add_argument("--res_dir", type=str, help="Path to the results directory", default="Results")

    seed = 1234
    fix_seed(seed)

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            setattr(args, k, v)
    else:
        args_file = os.path.join(args.res_dir, "config.txt")
        with open(args_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    if not os.path.exists(args.res_dir):
        os.mkdir(args.res_dir)

    # accepts data in cells x genes format; assumes that the last column contains labels for target cells (labels used only for downstream tasks)
    in_file = os.path.join(args.data_dir, args.fname + ".csv")
    ds = make_dataset(in_file=in_file, n_pca=args.n_comp_PCA, normalise=args.normalise)

    #train model
    embedding = train_model(args, ds, seed=seed)

    #visualise and store embedding
    col_palette = get_col_palette(ds.target.astype(str).values)
    col_dict = plotPoincareDisc(np.transpose(embedding), ds.target.astype(str).values, args.res_dir + args.fname, color_dict=col_palette)
    root_hat = poincare_root(args.root, ds.target.astype(str).values, ds.data.values)
    print('Root:', root_hat)
    titlename = '{0} rotated'.format(args.fname)
    poincare_coord_new = poincare_translation(-embedding[root_hat, :], embedding)

    plot_poincare_disc(poincare_coord_new, labels=ds.target.astype(str).values, coldict=col_dict,
                       file_name=args.res_dir + args.fname + '_rotated', d1=9.5, d2=9.0)

    pc_new_emb = pd.DataFrame(poincare_coord_new)
    pc_new_emb.to_csv(args.res_dir + args.fname + "_emb_rot.csv", index=False, header=False)



