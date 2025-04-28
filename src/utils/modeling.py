import torch
import torch.nn as nn
from transformers import AutoModel, BertModel


class stance_classifier(nn.Module):         #stance_classifier类继承自nn.Module（PyTorch中所有神经网络模型的基类）

    def __init__(self, num_labels, model_select, dropout):

        super(stance_classifier,self).__init()

        # 网络层
        self.dropout = nn.Dropout(dropout)  # dropout层，防止过拟合
        self.relu = nn.ReLU()               # ReLU层，激活函数

        # 模型选择，加载预训练模型实例
        if model_select == 'Bertweet':
            self.bert = AutoModel.from_pretrained("vinai/bertweet-base")    # 加载BERTweet预训练模型
        elif model_select == 'Bert':
            self.bert = BertModel.from_pretrained("bert-base-uncased")      # 加载BERT模型

        # 全连接层
        # self.bert.config.hidden_size BERT隐藏层的大小，768（基础BERT）1024（大型BERT）

        # 将BERT隐藏层大小映射到相同的大小
        # 对BERT输出进行非线性变换，更好适应下游任务
        self.linear = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        # 两个BERT输出拼接映射到BERT的隐藏层大小
        # 处理两个输入文本的联合表示，把拼接结果表示为固定大小的向量
        self.linear2 = nn.Linear(self.bert.config.hidden_size*2, self.bert.config.hidden_size)

        # 输出层
        self.out = nn.Linear(self.bert.config.hidden_size, num_labels)  # 主任务的分类输出，隐藏层大小映射到标签空间
        self.out2 = nn.Linear(self.bert.config.hidden_size, 2)          # 辅助任务的分类输出，隐藏层大小映射到辅助任务的标签数量，2，用于二分类任务

    # 前向传播
    def forward(self,x_input_ids,x_seg_ids,x_atten_masks,x_len,x_input_ids2):

        # BERT模型的输出。用BERT对输入文本进行编码，获得隐藏层输出
        last_hidden = self.bert(input_ids = x_input_ids,attention_mask = x_atten_masks,token_type_ids = x_seg_ids)
        last_hidden2 = self.bert(input_ids = x_input_ids2,attention_mask = x_atten_masks,token_type_ids=x_seg_ids)

        # 提取每个输入文本的[CLS]token的隐藏状态，作为句子的表示
        query = last_hidden[0][:,0]
        query2 = last_hidden2[0][:,0]

        # 对句子的表示应用dropout，防止过拟合
        query = self.dropout(query)
        query2 = self.dropout(query2)

        # 拼接句子的表示，形成上下文向量
        context_vec = torch.cat((query,query2),dim = 1)


        linear = self.relu(self.linear(query))  # 主任务的句子表示进行 全连接和激活
        out = self.out(linear)                  # 获取主任务的输出

        linear2 = self.relu(self.linear2(context_vec))  # 上下文向量进行 全连接和激活
        out2 = self.out2(linear2)                       # 获取辅助任务的输出

        return out,out2