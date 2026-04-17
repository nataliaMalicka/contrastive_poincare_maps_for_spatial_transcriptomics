import pandas as pd
#import cudf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import Dataset
#import dask.dataframe as dd
from sklearn.decomposition import PCA


class make_dataset(Dataset):
    def __init__(self, in_file, n_pca=None, normalise=False):
        df = pd.read_csv(in_file, sep=',')
        self.n = len(df.columns)
        self.data = df.iloc[:, :-1]   #assumes that the last column is your target
        self.target = df.iloc[:, -1]
        self.col_names = self.data.columns
        self.proc_data = self.standardise(self.data, n_pca, normalise)

    def __getitem__(self, index):
        # the dataset must return a pair of samples: the anchor and a random one from the
        # dataset that will be used to corrupt the anchor
        random_idx = np.random.randint(0, len(self))
        random_sample = torch.tensor(self.proc_data[random_idx], dtype=torch.float)
        sample = torch.tensor(self.proc_data[index], dtype=torch.float)
        # modified by MYZ
        return sample, sample, index# sample, random_sample MYSZ
        #return sample, random_sample

    def __len__(self):
        return len(self.proc_data)

    #def to_dataframe(self):
    #    return pd.DataFrame(self.data, columns=self.columns)

    @property
    def shape(self):
        return self.proc_data.shape

    def standardise(self, sample, n_pc, normalise):

        proc_data = sample.to_numpy()

        if normalise:
            print("Standardising...")
            scaler = StandardScaler()
            proc_data = scaler.fit_transform(proc_data)

        if n_pc is not None:
            print("Computing principal components...")
            pca = PCA(n_components=n_pc)
            proc_data = pca.fit_transform(proc_data)

        return proc_data

    def train_test_split(self):
        #placeholder for future functionality
        return None