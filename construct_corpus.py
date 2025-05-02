import re
import random
import os
import json
os.environ["HF_DATASETS_OFFLINE"] = "1"
from datasets import load_dataset

AVG_LEN = 4
THINK_PROB = 0.1
THINK_START = '\x17'
THINK_END = '\x19'
OTHERS_PROB = 0.03

output_file = 'out.jsonl'

def custom_split(text):
    # 定义正则表达式
    # 第一种情况：前一个字符不是控制字符且不是空格，后面第一个字符是空格
    pattern_1 = r'(?<=[^\s\W])(?=\s)'
    # 第二种情况：前一个字符是换行符，后一个字符不是控制字符且不是空格
    pattern_2 = r'(?<=\n)(?=[^\s\W])'
    # 合并两个正则表达式，用 "|" 表示或关系
    combined_pattern = f'{pattern_1}|{pattern_2}'
    # 使用 re.split 按照正则表达式分割字符串
    result = re.split(combined_pattern, text)
    # 返回结果列表
    return result

def gen_one_cot(text_list):
    l = len(text_list)
    u = int(random.expovariate(1/AVG_LEN)) + 1
    if 2*u > l or l <= 3 or random.random() < OTHERS_PROB:
        u = 0
    s = random.randrange(0, l-u)
    thought = text_list[s:s+u]
    if random.random() < OTHERS_PROB:
        random.shuffle(thought)
    return THINK_START + ''.join(thought) + THINK_END

def cot(text_list):
    cot_list = []
    for t in text_list:
        if random.random() < THINK_PROB:
            cot_list.append(gen_one_cot(text_list))
        cot_list.append(t)
    return ''.join(cot_list)

def add_cot_for_text(s):
    return cot(custom_split(s))

dataset = load_dataset('json', data_files="some_file.jsonl")

with open(output_file, 'a+', encoding='utf8') as fp:
    for i in range(50000):
        text = dataset['train'][i]['text']
        text_cot = add_cot_for_text(text)
        text_json = json.dumps({"text": text_cot}, ensure_ascii=False)
        fp.write(text_json + '\n')
