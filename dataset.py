from torch.utils.data.dataset import Dataset
import torch
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class SMDDataset(Dataset):
    def __init__(self, max_length, step, filepath=None, examples=None, device=torch.device("cpu"), normalize=True):
        if filepath is None and examples is None:
            raise ValueError
        self.device = device
        if examples is not None:
            self.examples = examples
        else:
            features = self.get_features(filepath, normalize)
            self.examples = self.get_examples(features, max_length, step)

    def get_valid_examples(self, valid_portation, shuffle=True):
        if shuffle:
            random.shuffle(self.examples)
        valid_index = int(len(self.examples)*valid_portation)
        ret_examples = self.examples[:valid_index]
        self.examples = self.examples[valid_index:]
        return ret_examples

    def get_features(self, filepath, normalize=True):
        features = []
        with open(filepath) as f:
            for line in f:
                line = line.strip().split(",")
                features.append(list(map(float, line)))
        if normalize:
            scaler = MinMaxScaler()
            features = scaler.fit_transform(np.array(features)).tolist()
        return features

    def get_examples(self, features, max_length, step):
        examples = []
        total_len = len(features)
        start_index = 0
        end_index = 0
        while end_index < total_len:
            end_index = start_index + max_length + 1
            if end_index>total_len:
                end_index = total_len
                start_index = total_len - max_length
            start_index += step
            examples.append(features[start_index:end_index])
        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i)->torch.tensor:
        return torch.tensor(self.examples[i], dtype=torch.float).to(self.device)

class SMDDatasetTest(SMDDataset):
    def __init__(self, filepath, labelfilepath, max_length, step, device):
        super().__init__(max_length=max_length, step=step, filepath=filepath, device=device)

        self.labels = self.get_labels(labelfilepath)[-len(self.examples):]

    def get_labels(self, labelfilepath):
        labels = []
        with open(labelfilepath) as f:
            for line in f:
                labels.append(int(float(line.strip())))
        return labels

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.float).to(self.device), torch.tensor(self.labels[i], dtype=torch.float).to(self.device)

