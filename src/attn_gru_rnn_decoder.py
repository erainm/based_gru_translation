# Created by erainm on 2025/9/30 14:42.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：attn_gru_rnn_decoder
# @Description: 基于GRU与Attention构建解码器
# TODO:
import torch
import torch.nn as nn

from src.config import Config

conf = Config()

class Attn_GRU_RNN_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1, max_length=conf.MAX_LENGTH):
        super(Attn_GRU_RNN_Decoder,self).__init__()
        self.input_size=input_size      # 解码器 词嵌入层单词数 eg：4345
        self.hidden_size=hidden_size    # 解码器 词嵌入层每个单词的特征数
        self.dropout_p=dropout_p        # 置零比率，默认0.1
        self.max_length=max_length      # 最大长度10

        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=self.input_size, embedding_dim=self.hidden_size)

        # 定义线性层1：求q的注意力权重分布
        # 查询张量Q: 解码器每个时间步的隐藏层输出或者是当前输入的x
        # 键张量K: 解码器上一个时间步的隐藏层输出
        # self.hidden_size * 2 = q + k
        self.attn = nn.Linear(in_features=self.hidden_size * 2, out_features=self.max_length)

        # 定义线性层2：q+注意力结果表示融合后，在按照指定维度输出
        # 值张量V:编码部分每个时间步输出结果组合而成
        self.attn_combine = nn.Linear(in_features=self.hidden_size * 2, out_features=self.hidden_size)

        # 定义dropout层
        self.dropout = nn.Dropout(p=self.dropout_p)

        # 定义gru层
        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, batch_first=True)

        # 定义out层 解码器按照类别进行输出(256,4345)
        self.out = nn.Linear(in_features=self.hidden_size, out_features=self.input_size)

        # 实例化softomax层 数值归一化 以便分类
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        # input代表q [1,1] 二维数据 hidden代表k [1,1,256] encoder_outputs代表v [1,10,256]

        # 数据经过词嵌入层
        # 数据形状 [1,1] --> [1,1,256]
        embedded = self.embedding(input)
        # 使用dropout进行随机丢弃，防止过拟合
        embedded = self.dropout(embedded)
        # 1 求查询张量q的注意力权重分布, attn_weights[1,1,10]
        attn_weights = torch.softmax(
            self.attn(torch.cat(tensors=(embedded, hidden), dim=-1)), dim=-1)

        # 2 求查询张量q的注意力结果表示 bmm运算, attn_applied[1,1,256]
        # [1,1,10], [1,10,256] ---> [1,1,256]
        attn_applied = torch.bmm(input=attn_weights, mat2=encoder_outputs)

        # 3 q 与 attn_applied 融合，[1,1,512]
        output = torch.cat(tensors=(embedded, attn_applied), dim=-1)
        # 再按照指定维度输出 output[1,1,256], gru层输入形状要求
        output = self.attn_combine(output)

        # 查询张量q的注意力结果表示 使用relu激活
        output = torch.relu(output)

        # 查询张量经过gru、softmax进行分类结果输出
        # 数据形状[1,1,256],[1,1,256] --> [1,1,256], [1,1,256]
        output, hidden = self.gru(output, hidden)

        # output经过全连接层 out+softmax层, 全连接层要求输入数据为二维数据
        # 数据形状[1,1,256]->[1,256]->[1,4345]
        output = self.softmax(self.out(output[:, 0, :]))

        # 返回解码器分类output[1,4345]，最后隐层张量hidden[1,1,256] 注意力权重张量attn_weights[1,1,10]
        return output, hidden, attn_weights