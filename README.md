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
