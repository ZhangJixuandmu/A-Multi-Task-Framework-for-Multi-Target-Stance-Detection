import torch
from torch.utils.data import TensorDataset, DataLoader
# from torch.utils.data import TensorDataset,DataLoader
# from sympy.physics.units import length
from transformers import BertweetTokenizer, AutoModel, BertTokenizer

#模型路径
model_path="vinai/bertweet-base"

try:
    # 加载分词器
    tokenizer = BertweetTokenizer.from_pretrained(model_path)

    #加载模型
    model = AutoModel.from_pretrained(model_path)

    #测试分词器和模型
    example_text = "Hello,BERTweet ❤️"
    tokenized_text = tokenizer(example_text,return_tensors="pt")
    output = model(**tokenized_text)

    print("模型和分词器加载成功，且能够运行！")

except Exception as e:
    print(f"加载失败：{e}")

#分词 Tokenization
def convert_data_to_ids(tokenizer,target,target2,text):
    # 初始化列表
    input_ids,seg_ids,attention_masks,sent_len = [],[],[],[]

    # 处理 target 数据集
    for tar,sent in zip(target,text):
        encoded_dict = tokenizer.encode_plus(
            tar,                            #目标文本
            sent,                           #原始输入文本
            add_special_tokens=True,        #确保特殊标记，比如[CLS]和[SEP]被添加
            max_length=128,                 #限制输入长度（文本太长就截断，太短就填充到这个长度）
            padding='max_length',           #填充的方式是补足到最大长度
            return_attention_mask=True      #让分词器返回注意力掩码
        )

        #分词完成后，返回的是一个encoded_dict字典，里面包含input_ids,token_type_ids等字段
        #逐个添加到初始化的列表中

        input_ids.append(encoded_dict['input_ids'])                 #分词后的数字序列
        seg_ids.append(encoded_dict['token_type_ids'])              #segment IDs
        attention_masks.append(encoded_dict['attention_mask'])      #注意力掩码
        sent_len.append(sum(encoded_dict['attention_mask']))        #注意力掩码的和，作为句子长度存储到sent_len

    # 处理 target2 数据集
    for tar,sent in zip(target2,text):
        encoded_dict = tokenizer.encode_plus(
            tar,
            sent,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            return_attention_mask=True,
        )

        input_ids.append(encoded_dict['input_ids'])
        seg_ids.append(encoded_dict['token_type_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        sent_len.append(sum(encoded_dict['attention_mask']))

    return input_ids,seg_ids,attention_masks,sent_len

#数据预处理
def data_helper_bert(x_train_all,x_val_all,x_test_all,main_task_name,model_select):

    print('Loading data')

    #解包train data
    x_train,y_train,x_train_target,y_train2,x_train_target2 = (
        x_train_all[0], # 训练数据的 输入文本
        x_train_all[1], # 训练数据的 标签
        x_train_all[2], # 训练数据的 目标文本
        x_train_all[3], # 训练数据的 辅助标签
        x_train_all[4]  # 训练数据的 辅助目标文本
    )

    x_val,y_val,x_val_target,y_val2,x_val_target2 = (
        x_val_all[0],x_val_all[1],x_val_all[2],
        x_val_all[3],x_val_all[4]
    )

    x_test,y_test,x_test_target,y_test2,x_test_target2 = (
        x_test_all[0],x_test_all[1],x_test_all[2],
        x_test_all[3],x_test_all[4]
    )

    # 输出最初数据信息
    # print("Length of the original x_train:%d, the sum is: %d"%(len(x_train),sum(y_train)))
    print(f"Length of the original x_train:{len(x_train)}, the sum is:{sum(y_train)}")
    print("Length of the original x_val:%d, the sum is: %d"%(len(x_val),sum(y_val)))
    print("Length of the original x_test: %d, the sum is: %d"%(len(x_test),sum(y_test)))

    # 选择分词器
    if model_select == 'Bertweet':
        tokenizer = BertweetTokenizer.from_pretrained("vinai/bertweet-base",normalization=True)
    elif model_select == 'Bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

    # 转换数据
    x_train_input_ids,x_train_seg_ids,x_train_atten_masks,x_train_len = \
        convert_data_to_ids(tokenizer,x_train_target,x_train_target2,x_train)

    x_val_input_ids,x_val_seg_ids,x_val_atten_masks,x_val_len = \
        convert_data_to_ids(tokenizer,x_val_target,x_val_target2,x_val)

    x_test_input_ids,x_test_seg_ids,x_test_atten_masks,x_test_len = \
        convert_data_to_ids(tokenizer,x_test_target,x_test_target2,x_test)

    # 数据重组
    x_train_all = [
        x_train_input_ids,
        x_train_seg_ids,
        x_train_atten_masks,
        y_train,
        x_train_len,
        y_train2
    ]
    x_val_all = [x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len,y_val2]
    x_test_all = [x_test_input_ids,x_test_seg_ids,x_test_atten_masks,y_test,x_test_len,y_test2]

    return x_train_all,x_val_all,x_test_all


def data_load(x_all, batch_size, train_mode):
    """
    处理输入数据x_all，转换为Pytorch张量
    :param x_all:
    :param batch_size:
    :param train_mode:
    :return:
    """
    if train_mode == "unified":
        half_y = int(len(x_all[3])/2)
        y = x_all[3]
        y2 = [
                 1 if y[half_y+i] == y[i]
                 else 0
              for i in range(len(y[:half_y]))
             ] * 2
    elif train_mode == "adhoc":
        y2 = [1 if x_all[3][i] == x_all[5][i] else 0 for i in range(len(x_all[3]))]

    # 数据分割 张量化
    half = int(len(x_all[0])/2) # 输入ID列表x_all[0]分为两部分

    x_input_ids = torch.tensor(x_all[0][:half], dtype = torch.long).cuda()  # 前半部分输入ID
    x_input_ids2 = torch.tensor(x_all[0][half:], dtype = torch.long).cuda() # 后半部分

    x_seg_ids = torch.tensor(x_all[1], dtype = torch.long).cuda()           # 段落ID的张量，并移到GPU
    x_atten_masks = torch.tensor(x_all[2], dtype = torch.long).cuda()       # 注意力掩码的张量
    y = torch.tensor(x_all[3], dtype = torch.long).cuda()                   # 标签的张量
    x_len = torch.tensor(x_all[4], dtype = torch.long).cuda()               # 句子长度的张量
    y2 = torch.tensor(y2, dtype = torch.long).cuda()                        # 新生成标签的张量

    # TensorDataset 把张量打包成数据集
    tensor_loader = TensorDataset(x_input_ids,x_seg_ids,x_atten_masks,y,x_len,x_input_ids2,y2)

    # DataLoader 加载数据集 shuffle 数据迭代 batch_size 批量划分
    data_loader = DataLoader(tensor_loader,shuffle = True, batch_size = batch_size)

    # 返回张量和data_loader
    return x_input_ids, x_seg_ids, x_atten_masks, y, x_len, data_loader, x_input_ids2, y2


def sep_test_set(input_data):
    """
    功能：把input_data划分成多个子数据集
    :param input_data:
    :return:
    """
    data_list = [input_data[:355],input_data[890:1245],input_data[355:618],
                 input_data[1245:1508],input_data[618:890],input_data[1508:1780]
                 ]

    return data_list



