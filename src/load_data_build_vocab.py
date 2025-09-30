# Created by erainm on 2025/9/29 21:44.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：load_data_build_vocab
# @Description: 加载数据到内存,并构建英文词表和法文词表
# TODO:
from config import Config
from text_clean_func import normalizeString

conf = Config()

def get_data(data_path_file):
    # 按行读取文件
    with open(data_path_file, 'r', encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    print('lines ---> ', len(lines))

    # 按行清洗文本,构建语言对 my_pairs
    # 格式: [['英文句子', '法文句子'],[['英文句子', '法文句子']]
    tmp_pair, my_pairs = [], []
    for line in lines:
        for s in  line.split('\t'):
            tmp_pair.append(normalizeString(s))
        my_pairs.append(tmp_pair)
        # 清空tmp_pair,用于存储下一个句子的英语和法语的句子对
        tmp_pair = []
    print("len(my_pairs) ---> ", len(my_pairs))
    # 打印前4条数据
    # print("my_pairs前四条数据:", my_pairs[:4])
    # 打印第8000条的英文 法文数据
    # print("my_pairs[8000][0]--->", my_pairs[8000][0])
    # print("my_pairs[8000][1]--->", my_pairs[8000][1])

    # 3. 遍历语言对,构建英文单词字典、法文单词字典
    english_word2index = {'SOS':0, 'EOS':1}
    # 第三个单词下标值从2开始
    english_word_n = 2
    french_word2index = {'SOS':0, 'EOS':1}
    french_word_n = 2
    # 遍历语言对,获取英文单词字典、法文单词字典
    for pair in my_pairs:
        for word in pair[0].split(' '):
            if word not in english_word2index:
                english_word2index[word] = english_word_n
                # 更新下一个单词的下标
                english_word_n += 1
        for word in pair[1].split(' '):
            if word not in french_word2index:
                french_word2index[word] = french_word_n
                # 更新法语下一个单词的下标
                french_word_n += 1
    # english_index2word french_index2word {下标1:单词1, 下标2:单词2 ……}
    english_index2word = {v:k for k,v in english_word2index.items()}
    french_index2word = {v:k for k,v in french_word2index.items()}
    # print("english_index2word", english_index2word[10])
    # print("french_index2word", french_index2word[10])
    # print("len(english_word2index)-->", len(english_word2index))
    # print("len(french_word2index)-->", len(french_word2index))
    # print("english_word_n--->", english_word_n, "french_word_n-->", french_word_n)

    return (english_word2index,
            english_index2word,
            english_word_n,
            french_word2index,
            french_index2word,
            french_word_n,
            my_pairs)


if __name__ == '__main__':
    data_path_file = conf.data_path_file
    (english_word2index,
     english_index2word,
     english_word_n,
     french_word2index,
     french_index2word,
     french_word_n,
     my_pairs) = get_data(data_path_file)