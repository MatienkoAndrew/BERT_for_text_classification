import torch
import transformers as ppb
import warnings
warnings.filterwarnings('ignore')

from torch import nn

import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import CONFIG
from datasets import DataPreparation
from models import BertClassifier
from training_utils import train, evaluate_on_test
        

def main():
    data_prep = DataPreparation(CONFIG)
    data_prep.prepare_dataloaders()
    train_loader, valid_loader, test_loader = data_prep.get_dataloaders()

    model = ppb.DistilBertModel.from_pretrained(CONFIG['pretrained_weights']).to(CONFIG['device'])
    bert_clf = BertClassifier(model).to(CONFIG['device'])
    optimizer = optim.Adam(bert_clf.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    train(bert_clf, train_loader, valid_loader, optimizer, criterion)

    best_model = BertClassifier(model).to(CONFIG['device'])
    best_model.load_state_dict(torch.load(CONFIG['path_to_save_model']))
    acc_test = evaluate_on_test(best_model, test_loader)
    print(f"Accuracy on test: {acc_test} | My trained model")

    # we have the same tokenizer
    # new_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    new_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(CONFIG['device'])
    acc_test = evaluate_on_test(new_model, test_loader)
    print(f"Accuracy on test: {acc_test} | Transformer model")

    return 0


if __name__ == '__main__':
    main()
