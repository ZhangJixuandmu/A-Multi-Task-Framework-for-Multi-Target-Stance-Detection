import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support


def compute_f1(preds, y):   #preds:模型预测值 y：真实标签

    rounded_preds = F.softmax(preds, dim = 1)   # 计算softmax
    _, indices = torch.max(rounded_preds, 1)    # 预测类别

    correct = (indices == y).float()    # 布尔值张量
    # correct = indices == y
    acc = correct.sum()/len(correct)

    y_pred = np.array(indices.cpu().numpy())    # 预测的标签类别，张量从GPU转移到CPU，转换为Numpy数组
    y_true = np.array(y.cpu().numpy())  # 真实的标签

    # 计算精度、召回率、F1分数、支持度
    result = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0,1,2])

    f1_average = (result[2][0]+result[2][2])/2  # 计算类别 0 和类别 2 的 F1 分数的平均值

    return acc,f1_average,result[0],result[1]