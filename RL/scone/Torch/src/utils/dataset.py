
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import numpy as np

# -------------------------------
# EXPERT DATASET
# -------------------------------
class ExpertDatasetFromSTO(Dataset):
    def __init__(self, sto_files, obs_cols, act_cols, skip_rows=6):
        """
        Expert dataset built from Scone .sto log files.

        Args:
            sto_files (list[str]): list of paths to .sto files
            obs_cols (list[str]): column names used as observations
            act_cols (list[str]): column names used as actions
            skip_rows (int): number of rows to skip (default: 6, for .sto headers)
        """
        self.obs_data = []
        self.act_data = []

        for file_path in sto_files:
            df = pd.read_csv(file_path, sep='\t', comment='%', skiprows=skip_rows)
            
            # Drop rows with NaN values
            df = df.dropna()

            # Extract observation and action arrays
            obs = df[obs_cols].values.astype(np.float32)
            act = df[act_cols].values.astype(np.float32)

            self.obs_data.append(obs)
            self.act_data.append(act)

        # Concatenate all episodes into single arrays
        self.obs_data = torch.tensor(np.vstack(self.obs_data))
        self.act_data = torch.tensor(np.vstack(self.act_data))

    def __len__(self):
        return len(self.obs_data)

    def __getitem__(self, idx):
        # Return a single (observation, action) pair
        return self.obs_data[idx], self.act_data[idx]