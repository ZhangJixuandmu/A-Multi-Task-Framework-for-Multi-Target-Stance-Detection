import re

import pandas as pd
import wordninja


def load_data(filename,usecols,col):

    filename = [filename]
    concat_text = pd.DataFrame
    raw_text = pd.read_csv(filename[0], usecols=[0],encoding='ISO-8859-1')
    raw_label = pd.read_csv(filename[0], usecols=[usecols[0]],encoding='ISO-8859-1')
    raw_target = pd.read_csv(filename[0], usecols=[usecols[1]],encoding='ISO-8859-1')

    if col == "Stance1":
        raw_label2 = pd.read_csv(filename[0],usecols=[4],encoding='ISO-8859-1')             # 辅助标签
        raw_target2 = pd.read_csv(filename[0],usecols=[3],encoding='ISO-8859-1')            # 辅助目标
    else:
        raw_label2 = pd.read_csv(filename[0],usecols=[2],encoding='ISO-8859-1')
        raw_target2 = pd.read_csv(filename[0],usecols=[1],encoding='ISO-8859-1')

    label = pd.DataFrame.replace(raw_label,['FAVOR','NONE','AGAINST'],[2,1,0])              # 标签数字化
    label2 = pd.DataFrame.replace(raw_label2,['FAVOR','NONE','AGAINST'],[2,1,0])
    concat_text = pd.concat([raw_text,label,raw_target,label2,raw_target2],axis=1)          # 拼接DataFrame,按列

    return concat_text

def data_clean(strings, norm_dict, clean_data):
    # 手动清理URL
    clean_data = re.sub(r'http\S+','',strings)
    # 删除表情符号
    clean_data = re.sub(r'[\x00-\x7F]+','',clean_data)
    # 删除保留字符
    clean_data = re.sub(r'[^a-zA-Z0-9#@,.!?&\<>=$\s]','',clean_data)
    # 删除特定的标签 #SemST
    clean_data = re.sub(r"#SemST", "", clean_data)
    # 用正则表达式提取单词、标点符号、数字等
    clean_data = re.findall(r"[A-Za-z#@]+|[,.!?&/\<>=$]|[0-9]+", clean_data)
    # 将每个词小写化
    clean_data = [[x.lower()] for x in clean_data]

    # 如果词汇在norm_dict中，替换为规范化的值
    for i in range(len(clean_data)):
        if clean_data[i][0] in norm_dict.keys():
            clean_data[i][0] = norm_dict[clean_data[i][0]]
        elif clean_data[i][0].startswith('#') or clean_data[i][0].startswith("@"):
            clean_data[i] = wordninja.split(clean_data[i][0])

    # 扁平化列表
    clean_data = [j for i in clean_data for j in i]

    return clean_data

def clean_all(filename,col,norm_dict):

    if col == "Stance1":
        usecols = [2,1]
    else:
        usecols = [4,3]

    concat_text = load_data(filename,usecols,col)
    raw_data = concat_text['Tweet'].values.tolist()                 # 提取’Tweet‘列数据并转换为列表

    if col == "Stance1":
        label = concat_text['Stance 1'].values.tolist()
        x_target = concat_text['Target 1'].values.tolist()

        label2 = concat_text['Stance 2'].values.tolist()
        x_target2 = concat_text['Target 2'].values.tolist()
    else:
        label = concat_text['Stance 2'].values.tolist()
        x_target = concat_text['Target 2'].values.tolist()

        label2 = concat_text['Stance 1'].values.tolist()
        x_target2 = concat_text['Target 1'].values.tolist()

    clean_data = [None for _ in range(len(raw_data))]

    for i in range(len(raw_data)):
        clean_data[i] = data_clean(raw_data[i],norm_dict, clean_data)
        x_target[i] = data_clean(x_target[i],norm_dict, clean_data)
        x_target2[i] = data_clean(x_target2,norm_dict, clean_data)

    return clean_data,label,x_target,label2,x_target2