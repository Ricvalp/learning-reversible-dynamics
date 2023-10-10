import numpy as np
import torch.utils.data as data
from matplotlib.colors import to_rgb


class Dataset(data.Dataset):
    def __init__(self, train_lines, num_lines, u0_path, T_path, train=True):
        super().__init__()
        self.train_lines = train_lines
        self.num_lines = num_lines
        self.u0_path = u0_path
        self.T_path = T_path
        self.train = train
        self.generate_data()

    def generate_data(self):
        data_input = []
        data_output = []

        if self.train:
            with open(self.u0_path) as f:
                for _ in range(self.train_lines):
                    data_input.append([float(f.readline()), float(f.readline())])

            with open(self.T_path) as f:
                for _ in range(self.train_lines):
                    data_output.append([float(f.readline()), float(f.readline())])

            self.data = np.array(data_input)
            self.label = np.array(data_output)

        else:
            with open(self.u0_path) as f:
                for _ in range(self.train_lines):
                    float(f.readline())
                    float(f.readline())
                for _ in range(self.num_lines - self.train_lines):
                    data_input.append([float(f.readline()), float(f.readline())])

            with open(self.T_path) as f:
                for _ in range(self.train_lines):
                    float(f.readline())
                    float(f.readline())
                for _ in range(self.num_lines - self.train_lines):
                    data_output.append([float(f.readline()), float(f.readline())])

            self.data = np.array(data_input)
            self.label = np.array(data_output)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        return data_point, data_label
