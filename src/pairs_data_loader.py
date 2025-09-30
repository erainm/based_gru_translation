# Created by erainm on 2025/9/30 13:17.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：pairs_data_loader
# @Description: 构建数据源对象及数据加载器
# TODO:
import torch
from torch.utils.data import Dataset

from src.config import Config

conf = Config()
class PairsDataset(Dataset):
    def __init__(self, my_pairs, english_word2index, french_word2index):
        # 样本x
        self.my_pairs = my_pairs
        self.english_word2index = english_word2index
        self.french_word2index = french_word2index
        # 样本条目
        self.sample_len = len(my_pairs)

    # 获取样本条数
    def __len__(self):
        return self.sample_len

    # 根据索引获取样本数据
    def __getitem__(self, index):
        # 对index异常值进行修正,防止越界
        index = min(max(index, 0), self.sample_len - 1)

        # 按索引获取数据
        x = self.my_pairs[index][0] # 英语句子
        y = self.my_pairs[index][1] # 法语句子

        # 对样本进行数值化
        x = [self.english_word2index[word] for word in x.split(' ')]
        x.append(conf.EOS_TOKEN)
        tensor_x = torch.tensor(x, dtype=torch.long, device=conf.device)
        # print('tensor_x.shape===>', tensor_x.shape, tensor_x)

        y = [self.french_word2index[word] for word in y.split(' ')]
        y.append(conf.EOS_TOKEN)
        tensor_y = torch.tensor(y, dtype=torch.long, device=conf.device)
        # 注意 tensor_x tensor_y都是一维数组，通过DataLoader拿出的数据是二维数据
        # print('tensor_y.shape===>', tensor_y.shape, tensor_y)

        # 返回结果
        return tensor_x, tensor_y