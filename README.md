# BERT_for_text_classification

This project aims to utilize the BERT model for the purpose of text classification. Given a dataset with reviews, we train a model to classify the sentiment of these reviews.

## Структура проекта

bash
```
├── README.md
├── [homework_part2]BERT_for_text_classification.ipynb # Jupyter notebook containing the project details and experimentation
├── __main__.py # Main script to run the project
├── config.py # Configuration file containing model and training parameters
├── datasets.py # Contains utilities to handle datasets
├── models # Directory to store model(s)
│   └── best-val-model.pt # Best model checkpoint
├── models.py # Model definitions
├── requirements.txt # List of required packages to run the project
└── training_utils.py # Utilities for training the model
```

## Использование

1. Убедитесь, что у вас установлены все необходимые библиотеки из файла requirements.txt. Вы можете установить их, выполнив следующую команду:

```
pip install -r requirements.txt
```

2. Запустите файл main.py, чтобы начать обучение и генерацию текста:

```
python __main__.py
```

3. Вывод

```
Training...:   3%|███▏                                                                                                   | 173/5536 [01:10<36:29,  2.45it/s]
  3%|███▊                                                                                                                  | 22/692 [00:02<01:20,  8.32it/s]
Epoch: 01
        Train Loss: 0.384 | Train PPL:   1.468
         Val. Loss: 0.314 |  Val. PPL:   1.368
Training...:   3%|███▏                                                                                                   | 173/5536 [01:11<36:57,  2.42it/s]
  3%|███▊                                                                                                                  | 22/692 [00:02<01:17,  8.66it/s]
Epoch: 02
        Train Loss: 0.195 | Train PPL:   1.215
         Val. Loss: 0.338 |  Val. PPL:   1.402
  3%|███▊                                                                                                                  | 22/692 [00:02<01:16,  8.71it/s]
Accuracy on test: 0.8771676300578035 | My trained model
  3%|███▊                                                                                                                  | 22/692 [00:02<01:15,  8.87it/s]
Accuracy on test: 0.9869942196531792 | Transformer model
```