# Created by erainm on 2025/9/30 13:33.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：unit_test
# @Description: 单元测试
# TODO:
from torch.utils.data import DataLoader

from src.attn_tensor_mapping import attention_tensor_mapping
from src.config import Config
from src.evaluate_model import dm_test_seq2seq_evaluate
from src.gru_rnn_encoder import GRU_RNN_Encoder
from src.load_data_build_vocab import get_data
from src.pairs_data_loader import PairsDataset
from src.train_model import train_seq2seq

conf = Config()

def test_PairsDataset():

    # 1 调用my_getdata函数获取数据
    (
        english_word2index,
        english_index2word,
        english_word_n,
        french_word2index,
        french_index2word,
        french_word_n,
        my_pairs,
    ) = get_data(conf.data_path_file)
    print("english_index2word--->",english_index2word)

    # 实例化dataset对象
    pairsdataset = PairsDataset(my_pairs, english_word2index, french_word2index)

    # 实例化DataLoader
    dataloader = DataLoader(pairsdataset, batch_size=1, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print("x.shape", x.shape, x)
        print("y.shape", y.shape, y)
        break

def test_gru_rnn_encoder():
    # 调用my_getdata函数获取数据
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
    gru_rnn_encoder = GRU_RNN_Encoder(english_word_n, hidden_size=256).to(conf.device)
    print("my_encoderrnn模型结构--->", gru_rnn_encoder)

    # 将数据喂给模型
    for x, y in mydataloader:
        print("x.shape", x.shape, x)
        print("y.shape", y.shape, y)

        # encode_output_c: 未加attention的中间语义张量c
        encode_output_c, hn = gru_rnn_encoder(input=x, h0=gru_rnn_encoder.inithidden())
        print("encode_output_c.shape--->", encode_output_c.shape, encode_output_c)


if __name__ == '__main__':
    # test_PairsDataset()
    # test_gru_rnn_encoder()
    # train_seq2seq()
    # dm_test_seq2seq_evaluate()
    attention_tensor_mapping()