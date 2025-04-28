import torch
import torch.nn as nn
import argparse
import json
import random
import numpy as np
from transformers import AdamW

import utils.preprocessing as pp
import utils.data_helper as dh
from src.utils import modeling, model_eval
from src.utils.data_helper import output


def run_classifier():

    parser = argparse.ArgumentParser(description="Stance Classifier Training Script")

    parser.add_argument("--input_target",type=str,default="trump_hillary",help="Specify the target topic(e.g., 'trump_hillary')")  # 指定目标主题
    parser.add_argument("--model_select",type=str,default="Bertweet",help="Select the model to use: 'BERTweet' or 'BERT'") # 模型选择
    parser.add_argument("--col",type=str,default="Stance1",help="Specify the column to use : 'Stance1' or 'Stance2'")   # 指定数据集特定列
    parser.add_argument("--train_mode",type=str,default="adhoc",help="Specify the training mode: 'unified' or 'adhoc' ")    # 训练模式
    parser.add_argument("--lr",type=float,default=2e-5,help="Set the learning rate for the optimizer") # 学习率
    parser.add_argument("--batch_size",type=int,default=32,help="Set the batch size for training") # 每批样本数
    parser.add_argument("--epochs",type=int,default=20,help="Set the number of training epochs") # 训练总轮次
    parser.add_argument("--dropout",type=float,default=0,help="Set the dropout rate for the regularization")   # 随机失活率，防过拟合
    parser.add_argument("--alpha",type=float,default=0.5,help="Set the weighting parameter for multi-task")   # 多任务学习中任务权重参数
    args = parser.parse_args()  # 使用parse_args方法解析命令行参数，存args变量中
    # print(args)
    # Welcome message and parameters
    print("Welcome to the Stance Classifier Training Script!")
    print("\nYou are running the script with the following parameters:")
    print(f"  Input Target: {args.input_target}")
    print(f"  Model Selected: {args.model_select}")
    print(f"  Column: {args.col}")
    print(f"  Training Mode: {args.train_mode}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Dropout Rate: {args.dropout}")
    print(f"  Alpha (multi-task weight): {args.alpha}\n")
    print(args)

    # random seed,实验可重复
    random_seeds = [2,3,4,5,6,7]

    # 命令行参数值 赋给 变量
    target_word_pair = [args.input_target]
    model_select     = args.model_select
    col              = args.col
    train_mode       = args.train_mode
    lr               = args.lr
    batch_size       = args.batch_size
    total_epoch      = args.epochs
    dropout          = args.dropout
    alpha            = args.alpha

    # 加载两个词典 同于 文本标准化
    # JSON文件，包含非正式语言的映射
    with open("./noslang_data.json","r") as f:
        data1 = json.load(f)
    # TXT文件，包含额外的词映射
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    # 两个词典合并为一个 用于 文本预处理
    normalization_dict = {**data1,**data2}

    for target_index in range(len(target_word_pair)):  # 遍历每个目标词对
        best_result,best_val = [],[]

        for seed in random_seeds:                   # 遍历每个种子，确保每次实验都有不同的随机初始化
            print("current random seed: ",seed)

            if train_mode == "unified":
                filename1 = '../data/raw_train_all_onecol.csv'
                filename2 = '../data/raw_val_all_onecol.csv'
                filename3 = '../data/raw_test_all_onecol.csv'
                x_train, y_train, x_train_target, y_train2, x_train_target2 = pp.clean_all(filename1, 'Stance1', normalization_dict)
                x_val, y_val, x_val_target, y_val2, x_val_target2 = pp.clean_all(filename2, 'Stance1', normalization_dict)
                x_test, y_test, x_test_target, y_test2, x_test_target2 = pp.clean_all(filename3, 'Stance1', normalization_dict)
            elif train_mode =='adhoc':
                filename1 = '../data/raw_train_'+target_word_pair[target_index]+'.csv'
                filename2 = '../data/raw_val_'+target_word_pair[target_index]+'.csv'
                filename3 = '../data/raw_test_'+target_word_pair[target_index]+'.csv'
                x_train,y_train,x_train_target,y_train2,x_train_tartget2 = pp.clean_all(filename1,col,normalization_dict)
                x_val,y_val,x_val_target,y_val2,x_val_target2 = pp.clean_all(filename2,col,normalization_dict)
                x_test,y_test,x_test_target,y_test2,x_test_target2 = pp.clean_all(filename3,col,normalization_dict)

            num_labels = len(set(y_train))
            x_train_all = [x_train,y_train,x_train_target,y_train2,x_train_tartget2]
            x_val_all = [x_val,y_val,x_val_target,y_val2,x_val_target2]
            x_test_all = [x_test,y_test,x_test_target,y_test2,x_test_target2]

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            #prepare for model
            # 将数据转换成BERT模型输入的格式
            x_train_all,x_val_all,x_teat_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,target_word_pair[target_index],model_select)

            # 生成数据加载器trainloader valloader testloader
            x_train_input_ids,x_train_seg_ids,x_train_atten_masks,y_train,x_train_len,trainloader,x_train_input_ids2,y_train2 = dh.data_load(x_train_all,batch_size,train_mode)
            x_val_input_ids,x_val_seg_ids,x_val_atten_masks,y_val,x_val_len,valloader,x_val_input_ids2,y_val2 = dh.data_load(x_val_all,batch_size,train_mode)
            x_test_input_ids,x_test_seg_ids,x_test_atten_masks_y_test,x_test_len,testloader,x_test_input_ids2,y_test2 = dh.data_load(x_test_all,batch_size,train_mode)

            # 创建情感分类器Model 转移到GPU上加速
            model = modeling.stance_classifier(num_labels,model_select,dropout).cuda()

            for n,p in model.named_parameters():

                if "bert.embeddings" in n:  # 冻结BERT的embedding层参数，不进行更新（从大量数据中学到了很好的词向量表示）
                    p.requires_grad = False # 反向传播时不计算参数的梯度，节省计算资源

            # 为每层设置不同学习率
            optimizer_grouped_parameters = [
                {'params': [p for n,p in model.named_parameters() if n.startswith('bert.encoder')], 'lr':lr},      # 编码器层
                {'params': [p for n,p in model.named_parameters() if n.startswith('bert.pooler')],'lr': 1e-3},  #池化层
                {'params': [p for n,p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},  # 线性层
                {'params': [p for n,p in model.named_parameters() if n.startswith('out')], 'lr':1e-3}]  # 输出层

            # 损失函数 交叉熵损失函数 所有样本损失求和
            loss_function = nn.CrossEntropyLoss(reduction = 'sum')
            # 优化器 AdamW 接收分层参数，更新模型参数
            optimizer = AdamW(optimizer_grouped_parameters)

            # 初始化变量
            sum_loss = []   # 存储训练损失
            sum_val = []    # 存储验证损失，目前没用到
            val_f1_average = []
            if train_mode == "unified":
                test_f1_average = [[] for i in range(6)]
            elif train_mode == 'adhoc':
                test_f1_average = [[]]

            # model训练循环，遍历每个epoch
            for epoch in range(0,total_epoch):
                print('Epoch: ', epoch)

                train_loss , valid_loss = [],[]

                # 模型设置为 训练模式
                model.train()
                for input_ids,seg_ids,atten_masks,target,length,input_ids2,target2 in trainloader:
                    optimizer.zero_grad()   # 调用方法，清空之前计算的梯度

                    output1, output2 = model(input_ids, seg_ids, atten_masks, length, input_ids2)

                    loss1 = loss_function(output1, target)
                    loss2 = loss_function(output2, target2)
                    loss = loss1 +loss2 * alpha
                    loss.backward() # 计算梯度

                    nn.utils.clip_grad_norm_(model.named_parameters(), 1) # 对梯度裁剪，防止梯度爆炸
                    optimizer.step()    # 更新模型参数
                    train_loss.append(loss.item())  # 损失值添加到列表(train_loss)

                sum_loss.append(sum(train_loss)/len(x_train))   # 当前epoch的平均训练损失(sum_loss)
                print(sum_loss[epoch])


                # 模型设置为评估模式
                model.eval()
                with torch.no_grad():   # 上下文管理器禁用梯度计算，减少内存消耗
                    # 验证集数据 传入 模型 得到主任务预测结果pred1
                    pred1,_=model(x_val_input_ids,x_val_seg_ids,x_val_atten_masks,x_val_len,x_val_input_ids2)
                    acc,f1_average,precision,recall=model_eval.compute_f1(pred1,y_val)
                    val_f1_average.append(f1_average)

                # 处理测试集数据
                if train_mode == "unified":       # 分割为多个子集
                    x_test_len_list = dh.sep_test_set(x_test_len)
                    y_test_list = dh.sep_test_set(y_test)
                    x_test_input_ids_list = dh.sep_test_set(x_test_input_ids)
                    x_test_seg_ids_list = dh.sep_test_set(x_test_seg_ids)
                    x_test_atten_masks_list = dh.sep_test_set(x_test_atten_masks)
                    x_test_input_ids_list2 = dh.sep_test_set(x_test_input_ids2)

                elif train_mode == "adhoc":     # 封装成列表
                    x_test_len_list = [x_test_len]
                    y_test_list = [y_test]
                    x_test_input_ids_list, x_test_seg_ids_list, x_test_atten_masks_list, x_test_input_ids_list2 = [x_test_input_ids], [x_test_seg_ids], [x_test_atten_masks], [x_test_input_ids2]
                with torch.no_grad():
                    for ind in range(len(y_test_list)):
                        pred1,_ = model(x_test_input_ids_list[ind],x_test_seg_ids_list[ind],x_test_atten_masks_list[ind],x_test_len_list[ind],x_test_input_ids_list2[ind])
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_test_list[ind])
                        test_f1_average[ind].append(f1_average)

            # 验证集F1分数均值列表中最大值 对应的 最后一个索引(epoch) 存储在 best_epoch 变量种
            best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1]
            # 根据best_epoch 提取 对应结果 添加到 best_result 列表中
            best_result.append([f1[best_epoch] for f1 in test_f1_average])


            print("******************************************")
            print("dev results with seed {} on all epoches".format(seed))
            print(val_f1_average)
            best_val.append(val_f1_average[best_epoch])

            print("******************************************")
            print("test results with seed {} on all epochs".format(seed))
            print(test_f1_average)
            print("******************************************")

            print("model performance on the test set: ")
            print(best_result)

if __name__ == "__main__":
    run_classifier()









        
