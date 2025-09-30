# Created by erainm on 2025/9/29 21:30.
# IDE：PyCharm 
# @Project: based_gru_translation
# @File：text_clean_func
# @Description: 文本清洗函数
# TODO:
import re


def normalizeString(s: str):
    s = s.lower().strip()
    # print('s1 ---> ', s)
    s = re.sub(r"([.!?])", r" \1", s)
    # print('s2 ---> ', s)
    s = re.sub(r"[^a-z.!?]+", r" ", s)
    # print('s3 ---> ', s)
    return s

if __name__ == '__main__':
    normalizeString("I am a boy@. I am a man.\n")