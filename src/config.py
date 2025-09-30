# Created by erainm on 2025/9/29 21:29.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：config
# @Description: 配置文件
# TODO:

import torch

class Config(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 起始标志
        self.SOS_TOKEN = 0
        # 结束标志
        self.EOS_TOKEN = 1
        # 句子最大长度
        self.MAX_LENGTH = 10
        # 数据文件位置
        self.data_path_file = '../data/eng-fra-v2.txt'
        # 模型训练参数
        self.mylr = 1e-4
        self.epochs = 2
        self.print_interval_num = 1000
        self.plot_interval_num = 100
        # 模型保存位置
        self.PATH1 = "../model/my_encoderrnn_2.pth"
        self.PATH2 = "../model/my_attndecoderrnn_2.pth"