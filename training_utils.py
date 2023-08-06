import torch
from config import CONFIG
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

import numpy as np
import math


def train(model, train_loader, valid_loader, optimizer, criterion, clip: int=1):
    """
    Train the model using the provided training and validation data.

    The function performs training over a number of epochs specified in the CONFIG global variable.
    After each epoch, the model's performance is evaluated on the validation dataset using the 
    evaluate_on_valid function. If the validation performance is improved, the model's state is saved.
    Additionally, the gradients are clipped to prevent exploding gradients.
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training set.
    - valid_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - optimizer (torch.optim.Optimizer): Optimizer for model training.
    - criterion (torch.nn.Module): The loss function used for training.
    - clip (int, optional): Maximum allowed norm for gradients. Defaults to 1.
    
    Returns:
    - None. But updates model's weights, and potentially saves the model state to a file.

    Note:
    - The function assumes that the model's forward method returns a dictionary with the 
      key 'logits' mapping to the model's output logits.
    - The training process can be visualized with a progress bar thanks to the tqdm function.
    - This function uses a global CONFIG variable for settings like the number of epochs 
      and the path to save the model.
    - Gradients are clipped after the backward pass and before the optimizer step to prevent 
      them from becoming too large.
    """
    train_history = []
    valid_history = []
    best_valid_loss = float('inf')
    for epoch in range(CONFIG['n_epochs']):
        model.train()
        
        train_loss = 0
        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc='Training...'):
            inputs = batch['inputs'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            labels = labels.long()
            attention_mask = batch['attention_mask'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            output = model(inputs, attention_mask=attention_mask)['logits']

            loss = criterion(output, labels) 
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss += loss.item()
        train_loss /= (i + 1)
            
        valid_loss = evaluate_on_valid(model, valid_loader, criterion)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), CONFIG['path_to_save_model'])
        
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        print(f'Epoch: {epoch+1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


def evaluate_on_valid(model, valid_loader, criterion):
    """
    Evaluate the model's performance on the validation set.
    
    The function calculates the average loss over the validation dataset using a provided criterion.
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - valid_loader (torch.utils.data.DataLoader): DataLoader for the validation set.
    - criterion (torch.nn.Module): The loss function used for evaluation.
    
    Returns:
    - float: The average validation loss.
    
    Note:
    - The function assumes that the model's forward method returns a dictionary with the 
      key 'logits' mapping to the model's output logits.
    - It also uses a global CONFIG variable for device configurations.
    """
    model.eval()
    valid_loss = 0
    with torch.no_grad():
    
        for i, batch in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            inputs = batch['inputs'].to(CONFIG['device'])
            labels = batch['labels'].to(CONFIG['device'])
            labels = labels.long()
            attention_mask = batch['attention_mask'].to(CONFIG['device'])

            output = model(inputs, attention_mask=attention_mask)['logits']

            loss = criterion(output, labels)
            valid_loss += loss.item()
        
    return valid_loss / (i + 1)


def evaluate_on_test(model, test_loader):
    """
    Evaluate the model's accuracy on the test set.

    The function calculates the model's accuracy on the test set by comparing the predicted labels 
    with the true labels.
    
    Parameters:
    - model (torch.nn.Module): The PyTorch model to be evaluated.
    - test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
    
    Returns:
    - float: Accuracy of the model on the test set.

    Note:
    - The function assumes that the model's forward method returns a dictionary with the 
      key 'logits' mapping to the model's output logits.
    - The model's output logits are processed using a softmax operation to obtain the predicted class probabilities.
    - It also uses a global CONFIG variable for device configurations and requires the 
      accuracy_score function from sklearn.metrics to be imported for calculating accuracy.
    """
    pred_labels = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            inputs = batch['inputs'].to(CONFIG['device'])
            labels = batch['labels'].long().to(CONFIG['device'])
            attention_mask = batch['attention_mask'].to(CONFIG['device'])

            true_labels.append(labels.cpu().numpy())
            logits = model(inputs, attention_mask=attention_mask)['logits']
            probas = torch.softmax(logits, dim=-1)
            pred_labels_batch = torch.argmax(probas, dim=-1)
            pred_labels.append(pred_labels_batch.cpu().numpy())

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    accuracy = accuracy_score(true_labels, pred_labels)
    return accuracy
