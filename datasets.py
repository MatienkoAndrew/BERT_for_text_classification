from torch.utils.data import Dataset, random_split
from torch.utils.data import Sampler
from torch.utils.data import DataLoader

import transformers as ppb

import torch
import numpy as np
import pandas as pd


class ReviewsDataset(Dataset):
    """
    Initializes the ReviewsDataset with reviews, tokenizer, and labels.
    """
    def __init__(self, reviews, tokenizer, labels):
        self.labels = labels
        # tokenized reviews
        self.tokenized = [tokenizer.encode(review, add_special_tokens=True) for review in reviews]
        
    def __getitem__(self, idx):
        return {"tokenized": self.tokenized[idx], "label": self.labels[idx]}

    def __len__(self):
        return len(self.labels)


class ReviewsSampler(Sampler):
    """
    Класс ReviewsSampler предназначен для создания батчей 
    из датасета для обучения или тестирования модели. 
    Батчи сформированы таким образом, 
    что внутри одного батча находятся предложения схожей длины, 
    что позволяет уменьшить количество паддинга и ускорить процесс обучения.
    """
    def __init__(self, subset, batch_size=32):
        self.batch_size = batch_size
        self.subset = subset

        self.indices = subset.indices
        # tokenized for our data
        self.tokenized = np.array(subset.dataset.tokenized, dtype=object)[self.indices]

    def __iter__(self):

        batch_idx = []
        # index in sorted data
        for index in np.argsort(list(map(len, self.tokenized))):
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0:
            yield batch_idx

    def __len__(self):
        return len(self.subset)


class DataPreparation:
    """
    Prepares data for BERT training, validation, and testing.
    
    Attributes:
    - CONFIG (dict): Configuration dictionary containing settings and paths.
    - tokenizer: BERT tokenizer instance.
    - data (DataFrame): Loaded dataset.
    - train_data, valid_data, test_data: Split dataset instances.
    - train_loader, valid_loader, test_loader: Data loaders for the split datasets.
    
    Methods:
    - load_tokenizer(): Loads the BERT tokenizer based on CONFIG.
    - load_data(): Loads data from the path specified in CONFIG.
    - split_data(): Splits the loaded data into train, validation, and test datasets.
    - get_padded(values): Pads tokenized sequences to the same length.
    - collate_fn(batch): Combines multiple data points into a batch, and pads them.
    - prepare_dataloaders(): Initializes data loaders for train, validation, and test datasets.
    - get_dataloaders(): Returns the train, validation, and test data loaders.
    """
    def __init__(self, CONFIG) -> None:
        self.CONFIG = CONFIG
        self.tokenizer = self.load_tokenizer()
        self.data = self.load_data()
        self.split_data()

    def load_tokenizer(self):
        return ppb.DistilBertTokenizer.from_pretrained(self.CONFIG['pretrained_weights'])

    def load_data(self):
        return pd.read_csv(self.CONFIG['path_to_data'], delimiter='\t', header=None)

    def split_data(self):
        dataset = ReviewsDataset(self.data[0], self.tokenizer, self.data[1])
        train_size, val_size = int(.8 * len(dataset)), int(.1 * len(dataset))
        torch.manual_seed(self.CONFIG['SEED'])
        self.train_data, self.valid_data, self.test_data = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])

        # print(f"Number of training examples: {len(train_data)}")
        # print(f"Number of validation examples: {len(valid_data)}")
        # print(f"Number of testing examples: {len(test_data)}")

    def get_padded(self, values):
        max_len = 0
        for value in values:
            if len(value) > max_len:
                max_len = len(value)

        padded = np.array([value + [0]*(max_len-len(value)) for value in values])
        return padded

    def collate_fn(self, batch):
        inputs = []
        labels = []
        for elem in batch:
            inputs.append(elem['tokenized'])
            labels.append(elem['label'])

        inputs = self.get_padded(inputs)
        attention_mask = [[float(i>0) for i in seq] for seq in inputs]

        return {"inputs": torch.tensor(inputs), "labels": torch.FloatTensor(labels), 'attention_mask' : torch.tensor(attention_mask)}

    def prepare_dataloaders(self):
        self.train_loader = DataLoader(self.train_data, batch_sampler=ReviewsSampler(self.train_data), collate_fn=self.collate_fn)
        self.valid_loader = DataLoader(self.valid_data, batch_sampler=ReviewsSampler(self.valid_data), collate_fn=self.collate_fn)
        self.test_loader = DataLoader(self.test_data, batch_sampler=ReviewsSampler(self.test_data), collate_fn=self.collate_fn)

    def get_dataloaders(self):
        return self.train_loader, self.valid_loader, self.test_loader
