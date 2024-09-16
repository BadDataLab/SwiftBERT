import torch
import torch.nn as nn
import transformers
from transformers.models.bert.modeling_bert import BertModel


class ExitModel(nn.Module):
    def __init__(self):
        super(ExitModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear1 = nn.Linear(768, 128)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(.9)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, ids, mask):
        bert_output = self.bert(ids, attention_mask=mask)
        linear1_output = self.linear1(bert_output['last_hidden_state'][:, 0, :].view(-1, 768))
        relu_output = self.relu(linear1_output)
        dropout_output = self.dropout(relu_output)
        linear2_output = self.linear2(dropout_output)
        return linear2_output
