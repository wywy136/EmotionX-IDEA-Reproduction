from transformers import BertModel, BertForPreTraining
import Config
import torch.nn as nn


class IDEAModel(nn.Module):
    def __init__(self):
        super(IDEAModel, self).__init__()
        self._bert = BertModel.from_pretrained('/home/ramon/wy_uci/torch/save_pretrained_pretraining')
        self._linear = nn.Linear(768, 4)
        # self._out = nn.Linear(256, 4)
        self.dropout = nn.Dropout(Config.args.dropout_rate)

    def forward(self, source, mask, type):
        hidden = self._bert(source, mask, type)[0]
        # return self._out(F.relu(self._linear(self.dropout(hidden[:, 0, :]))))
        return self._linear(self.dropout(hidden[:, 0, :]))