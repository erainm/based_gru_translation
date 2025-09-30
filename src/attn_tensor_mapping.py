# Created by erainm on 2025/9/30 15:24.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：attn_tensor_mapping
# @Description: attention 张量制图
# TODO:
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.attn_gru_rnn_decoder import Attn_GRU_RNN_Decoder
from src.config import Config
from src.evaluate_model import seq2seq_evaluate
from src.gru_rnn_encoder import GRU_RNN_Encoder
from src.load_data_build_vocab import get_data
from src.pairs_data_loader import PairsDataset

conf = Config()
def attention_tensor_mapping():
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = get_data(conf.data_path_file)

    # 实例化dataset对象
    mypairsdataset = PairsDataset(my_pairs, english_word2index, french_word2index)
    # 实例化dataloader
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = GRU_RNN_Encoder(input_size, hidden_size).to(device=conf.device)
    # my_encoderrnn.load_state_dict(torch.load(PATH1))
    my_encoderrnn.load_state_dict(
        torch.load(conf.PATH1, map_location=lambda storage, loc: storage), False
    )

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = Attn_GRU_RNN_Decoder(input_size, hidden_size).to(device=conf.device)
    # my_attndecoderrnn.load_state_dict(torch.load(conf.PATH2))
    my_attndecoderrnn.load_state_dict(
        torch.load(conf.PATH2, map_location=lambda storage, loc: storage), False
    )

    sentence = "we re both teachers ."
    # 样本x 文本数值化
    tmpx = [english_word2index[word] for word in sentence.split(" ")]
    tmpx.append(conf.EOS_TOKEN)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=conf.device).view(1, -1)

    # 模型预测
    decoded_words, attentions = seq2seq_evaluate(
        tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word
    )
    # print("decoded_words->", decoded_words)

    # print('\n')
    # print('英文', sentence)
    # print('法文', output_sentence)

    # 创建热图
    fig, ax = plt.subplots()
    # cmap:指定一个颜色映射，将数据值映射到颜色
    # viridis:从深紫色（低值）过渡到黄色（高值），具有良好的对比度和可读性
    cax = ax.matshow(attentions[0].cpu().detach().numpy(), cmap="viridis")
    # 添加颜色条
    fig.colorbar(cax)
    # 添加标签
    for (i, j), value in np.ndenumerate(attentions[0].cpu().detach().numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white")

    # 确保图像保存目录存在
    img_dir = '../img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    # 保存图像
    plt.savefig("../img/s2s_attn.png")
    plt.show()

    print("attentions.numpy()--->\n", attentions.numpy())
    print("attentions.size--->", attentions.size())
