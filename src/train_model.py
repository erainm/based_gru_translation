# Created by erainm on 2025/9/30 14:59.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：train_model
# @Description: 模型训练
# TODO:
import time
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import random
import os
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from src.load_data_build_vocab import get_data

from src.attn_gru_rnn_decoder import Attn_GRU_RNN_Decoder
from src.config import Config
from src.gru_rnn_encoder import GRU_RNN_Encoder
from src.pairs_data_loader import PairsDataset

conf =Config()

def train_iters(
    x,
    y,
    my_encoderrnn: GRU_RNN_Encoder,
    my_attndecoderrnn: Attn_GRU_RNN_Decoder,
    myadam_encode,
    myadam_decode,
    mynllloss,
    total_steps,
    current_step,
):
    my_encoderrnn.train()
    my_attndecoderrnn.train()
    # 1 编码 encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)
    encode_hidden = my_encoderrnn.inithidden()
    encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)  # 一次性送数据
    # [1,6],[1,1,256] --> [1,6,256],[1,1,256]

    # 2 解码参数准备和解码
    # 解码参数1 encode_output_c [1, 10,256]
    encode_output_c = torch.zeros(
        1, conf.MAX_LENGTH, my_encoderrnn.hidden_size, device=conf.device
    )
    for idx in range(x.shape[1]):
        encode_output_c[:, idx, :] = encode_output[:, idx, :]

    # 解码参数2
    decode_hidden = encode_hidden

    # 解码参数3
    input_y = torch.tensor([[conf.SOS_TOKEN]], device=conf.device)

    myloss = 0.0
    iters_num = 0
    y_len = y.shape[1]

    # 教师强制机制, 阈值线性衰减
    teacher_forcing_ratio = max(0.1, 1 - (current_step / total_steps))
    # 阈值指数衰减
    # teacher_forcing_ratio = 0.9 ** current_step
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    for idx in range(y_len):
        # 数据形状 [1,1],[1,1,256],[1,10,256] ---> [1,4345],[1,1,256],[1,1,10]
        output_y, decode_hidden, attn_weight = my_attndecoderrnn(
            input_y, decode_hidden, encode_output_c
        )
        target_y = y[:, idx]
        myloss = myloss + mynllloss(output_y, target_y)
        iters_num += 1
        # 使用teacher_forcing
        if use_teacher_forcing:
            # 获取真实样本作为下一个输入
            input_y = y[:, idx].reshape(shape=(-1, 1))
        # 不使用teacher_forcing
        else:
            # 获取最大值的值和索引
            topv, topi = output_y.topk(1)
            if topi.item() == conf.EOS_TOKEN:
                break
            # 获取预测y值作为下一个输入
            input_y = topi.detach()

    # 梯度清零
    myadam_encode.zero_grad()
    myadam_decode.zero_grad()

    # 反向传播
    myloss.backward()

    # 梯度更新
    myadam_encode.step()
    myadam_decode.step()

    # 计算迭代次数的平均损失
    return myloss.item() / iters_num

def train_seq2seq():
    # 获取数据
    (english_word2index, english_index2word, english_word_n,
    french_word2index, french_index2word, french_word_n, my_pairs) = get_data(data_path_file=conf.data_path_file)
    # 实例化 mypairsdataset对象  实例化 mydataloader
    mypairsdataset = PairsDataset(my_pairs, english_word2index, french_word2index)
    mydataloader = DataLoader(dataset=mypairsdataset, batch_size=1, shuffle=True)

    # 实例化编码器 my_encoderrnn 实例化解码器 my_attndecoderrnn
    my_encoderrnn = GRU_RNN_Encoder(english_word_n, 256).to(conf.device)
    my_attndecoderrnn = Attn_GRU_RNN_Decoder(input_size=french_word_n, hidden_size=256, dropout_p=0.1, max_length=conf.MAX_LENGTH).to(conf.device)

    # 实例化编码器优化器 myadam_encode 实例化解码器优化器 myadam_decode
    myadam_encode = optim.Adam(my_encoderrnn.parameters(), lr=conf.mylr)
    myadam_decode = optim.Adam(my_attndecoderrnn.parameters(), lr=conf.mylr)

    # 实例化损失函数 mycrossentropyloss = nn.NLLLoss()
    mynllloss = nn.NLLLoss()

    # 定义模型训练的参数
    plot_loss_list = []
    # 统计所有轮次的总批次数
    total_steps = conf.epochs * len(mydataloader)
    # 当前累计批次数
    current_step = 0

    # 外层for循环 控制轮数 for epoch_idx in range(1, epochs + 1):
    for epoch_idx in range(1, conf.epochs + 1):
        print_loss_total, plot_loss_total = 0.0, 0.0
        starttime = time.time()

        # 内层for循环 控制迭代次数
        # start=1: 下标从1开始, 默认0, 数据开始从第1个开始取
        # item第1个值为1
        for item, (x, y) in enumerate(mydataloader, start=1):
            # 调用内部训练函数
            myloss = train_iters(x, y, my_encoderrnn, my_attndecoderrnn, myadam_encode, myadam_decode, mynllloss, total_steps, current_step)
            print_loss_total += myloss
            plot_loss_total += myloss
            # 累计训练批次数
            current_step += 1

            # 计算打印屏幕间隔损失-每隔1000次
            if item % conf.print_interval_num == 0:
                print_loss_avg = print_loss_total / conf.print_interval_num
                # 将总损失归0
                print_loss_total = 0
                # 打印日志，日志内容分别是：训练耗时，当前迭代步，当前进度百分比，当前平均损失
                print('轮次%d  损失%.6f 时间:%d' % (epoch_idx, print_loss_avg, time.time() - starttime))

            # 计算画图间隔损失-每隔100次
            if item % conf.plot_interval_num == 0:
                # 通过总损失除以间隔得到平均损失
                plot_loss_avg = plot_loss_total / conf.plot_interval_num
                # 将平均损失添加plot_loss_list列表中
                plot_loss_list.append(plot_loss_avg)
                # 总损失归0
                plot_loss_total = 0

        # 每个轮次保存模型
        # 确保模型保存目录存在
        model_dir = '../model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(my_encoderrnn.state_dict(), '../model/my_encoderrnn_%d.pth' % epoch_idx)
        torch.save(my_attndecoderrnn.state_dict(), '../model/my_attndecoderrnn_%d.pth' % epoch_idx)

    # 确保图像保存目录存在
    img_dir = '../img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # 所有轮次训练完毕 画损失图
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig('../img/s2sq_loss.png')
    plt.show()