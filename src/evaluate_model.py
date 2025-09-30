# Created by erainm on 2025/9/30 15:17.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：evaluate_model
# @Description: 模型评估
# TODO:
import torch
from src.load_data_build_vocab import get_data
from src.attn_gru_rnn_decoder import Attn_GRU_RNN_Decoder
from src.config import Config
from src.gru_rnn_encoder import GRU_RNN_Encoder
from src.pairs_data_loader import PairsDataset


conf = Config()
def seq2seq_evaluate(
    x, my_encoderrnn: GRU_RNN_Encoder, my_attndecoderrnn: Attn_GRU_RNN_Decoder, french_index2word
):
    with torch.no_grad():
        my_encoderrnn.eval()
        my_attndecoderrnn.eval()
        # 1 编码：一次性的送数据
        encode_hidden = my_encoderrnn.inithidden()
        encode_output, encode_hidden = my_encoderrnn(x, encode_hidden)

        # 2 解码参数准备
        # 解码参数1 固定长度中间语义张量c
        encoder_outputs_c = torch.zeros(
            1, conf.MAX_LENGTH, my_encoderrnn.hidden_size, device=conf.device
        )
        x_len = x.shape[1]
        for idx in range(x_len):
            encoder_outputs_c[:, idx, :] = encode_output[:, idx, :]

        # 解码参数2 最后1个隐藏层的输出 作为 解码器的第1个时间步隐藏层输入
        decode_hidden = encode_hidden

        # 解码参数3 解码器第一个时间步起始符
        input_y = torch.tensor([[conf.SOS_TOKEN]], device=conf.device)

        # 3 自回归方式解码
        # 初始化预测的词汇列表
        decoded_words = []
        # 初始化attention张量
        decoder_attentions = torch.zeros(1, conf.MAX_LENGTH, conf.MAX_LENGTH)
        for idx in range(conf.MAX_LENGTH):  # note:MAX_LENGTH=10
            output_y, decode_hidden, attn_weights = my_attndecoderrnn(
                input_y, decode_hidden, encoder_outputs_c
            )
            # 预测值作为下一次时间步的输入值
            topv, topi = output_y.topk(1)
            decoder_attentions[:, idx, :] = attn_weights[:, 0, :]

            # 如果输出值是终止符，则循环停止
            if topi.item() == conf.EOS_TOKEN:
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(french_index2word[topi.item()])

            # 将本次预测的索引赋值给 input_y，进行下一个时间步预测
            input_y = topi.detach()

    # 返回结果decoded_words，注意力张量权重分布表(把没有用到的部分切掉)
    # 句子长度最大是10, 长度不为10的句子的注意力张量其余位置为0, 去掉
    return decoded_words, decoder_attentions[:, :idx + 1, :]

def dm_test_seq2seq_evaluate():
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = get_data(data_path_file=conf.data_path_file)
    # 实例化dataset对象
    mypairsdataset = PairsDataset(my_pairs, english_word2index, french_word2index)

    # 实例化模型
    input_size = english_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_encoderrnn = GRU_RNN_Encoder(input_size, hidden_size).to(conf.device)

    """
    torch.load(map_location=)
    map_location: 指定如何重映射模型权重的存储设备（如 GPU → CPU 或 GPU → 其他 GPU）。
    # 加载到 CPU：map_location=torch.device('cpu') 或 map_location='cpu'。
    自动选择可用设备：map_location=torch.device('cuda')。
    自定义映射逻辑：通过函数定义设备映射规则。
    map_location=lambda storage, loc: storage -> 该lambda函数直接返回原始存储对象(storage)
    强制所有张量保留在保存时的设备上。当模型权重保存时的设备与当前环境一致时（例如均在CPU或同一GPU上），避免不必要的设备迁移。

    load_state_dict(strict=)
    strict:True（默认）:要求加载的权重键（keys）与当前模型的键完全匹配。如果存在不匹配（例如权重中缺少某些键，或模型有额外键），抛出RuntimeError。
    """
    my_encoderrnn.load_state_dict(
        torch.load(conf.PATH1, map_location=lambda storage, loc: storage), strict=False
    )
    print("my_encoderrnn模型结构--->", my_encoderrnn)

    # 实例化模型
    input_size = french_word_n
    hidden_size = 256  # 观察结果数据 可使用8
    my_attndecoderrnn = Attn_GRU_RNN_Decoder(input_size, hidden_size).to(conf.device)
    # my_attndecoderrnn.load_state_dict(torch.load(PATH2))
    my_attndecoderrnn.load_state_dict(
        torch.load(conf.PATH2, map_location=lambda storage, loc: storage), False
    )
    # print("my_decoderrnn模型结构--->", my_attndecoderrnn)

    my_samplepairs = [
        [
            "i m impressed with your french .",
            "je suis impressionne par votre francais .",
        ],
        ["i m more than a friend .", "je suis plus qu une amie ."],
        ["she is beautiful like her mother .", "elle est belle comme sa mere ."],
    ]
    # print("my_samplepairs--->", len(my_samplepairs))

    for index, pair in enumerate(my_samplepairs):
        x = pair[0]
        y = pair[1]

        # 样本x 文本数值化
        tmpx = [english_word2index[word] for word in x.split(" ")]
        tmpx.append(conf.EOS_TOKEN)
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=conf.device).view(1, -1)

        # 模型预测
        decoded_words, attentions = seq2seq_evaluate(
            tensor_x, my_encoderrnn, my_attndecoderrnn, french_index2word
        )
        # print("attentions--->", attentions)
        # print('decoded_words->', decoded_words)
        output_sentence = " ".join(decoded_words)

        print("\n")
        print(">", x)
        print("=", y)
        print("<", output_sentence)