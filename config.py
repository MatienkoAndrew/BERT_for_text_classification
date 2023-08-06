import torch

CONFIG = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'path_to_data': 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv',
    'pretrained_weights': 'distilbert-base-uncased',
    'batch_size': 64,
    'n_epochs': 2,
    'max_length': 128,
    'SEED': 42,
    'path_to_save_model': './models/best-val-model.pt',
    'dropout': 0.1
}
