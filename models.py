from torch import nn
from config import CONFIG
from transformers.modeling_outputs import SequenceClassifierOutput

class BertClassifier(nn.Module):
    """
    Custom classifier built on top of a pretrained BERT model.

    This classifier takes as input tokenized sequences and their attention masks, 
    passes them through a pretrained BERT model, and then uses a linear layer to produce 
    logits for binary classification (2 classes).

    Attributes:
    - bert (torch.nn.Module): The pretrained BERT model for feature extraction.
    - dropout (torch.nn.Dropout): Dropout layer for regularization.
    - relu (torch.nn.ReLU): Rectified Linear Unit activation function.
    - linear (torch.nn.Linear): Linear layer to produce logits from BERT's outputs.

    Methods:
    - forward(inputs, attention_mask): Computes logits for the given inputs and attention masks.

    Note:
    - The model assumes that the pretrained BERT model produces a last hidden state of shape 
      [batch_size, sequence_length, hidden_size], where hidden_size is 768 for the base BERT model.
    - Only the [CLS] token embedding (first token in the sequence) is used for classification after 
      passing through the dropout and linear layers.
    """
    def __init__(self, pretrained_model, dropout=CONFIG['dropout']):
        """
        Initializes the BertClassifier with a pretrained BERT model and dropout rate.

        Parameters:
        - pretrained_model (torch.nn.Module): Pretrained BERT model.
        - dropout (float, optional): Dropout rate for regularization. Defaults to 0.1.
        """
        super().__init__()

        self.bert = pretrained_model
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=768, out_features=2)
    
    def forward(self, inputs, attention_mask):
        """
        Computes logits for the given inputs and attention masks.

        Parameters:
        - inputs (torch.Tensor): Tokenized input sequences of shape [batch_size, sequence_length].
        - attention_mask (torch.Tensor): Binary tensor indicating the positions of padded tokens.

        Returns:
        - SequenceClassifierOutput: An object with a single attribute 'logits', containing the 
          computed logits of shape [batch_size, 2].
        """
        outputs = self.bert(inputs, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output) ##-- [batch_size, 2] - probability to be positive
        return SequenceClassifierOutput(logits=logits)
