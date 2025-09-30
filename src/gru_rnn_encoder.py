# Created by erainm on 2025/9/30 13:44.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：gru_rnn_encoder
# @Description: 构建基于GRU的编码器
# TODO:
import torch
import torch.nn as nn

from src.config import Config


class GRU_RNN_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRU_RNN_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 实例化Embedding层
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.hidden_size)

        # 实例化GRU层
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

    def forward(self, input, h0):
        embedded = self.embedding(input)
        # print("embedded", embedded.shape, embedded)
        # print("h0 ---> ", h0.shape, h0)
        # 数据经过gru
        output, hn = self.gru(embedded, h0)
        return output, hn

    def inithidden(self):
        # 将隐藏层张量初始化成为1x1xself.hidden_size大小的张量
        conf = Config()
        return torch.zeros(size=(1, 1, self.hidden_size), device=conf.device)