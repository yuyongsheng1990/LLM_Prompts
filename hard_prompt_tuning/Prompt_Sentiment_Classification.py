# -*- coding: utf-8 -*-
# @Time : 2024/6/26 11:46
# @Author : yysgz
# @File : Prompt_Sentiment_Classification.py
# @Project : LLM_Prompts
# @Description :
'''
数据集来自twitter 2013，数据集中有三种类别[positive, negative, neutral]，在预处理过程中我们去掉neutral类型数据。
在prompt-oriented fine-tuning任务中，我们构造一个prompt模板"it was [MASK]. sentence"将判断positive转换成完形填空预测good，将判断negative转换成完形填空预测bad。
在fine-tuning任务中，我们在预训练模型后加一层mlp，做二分类。
'''

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
from transformers import get_constant_schedule_with_warmup

from d2l import torch as d2l
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('WebAgg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# SVG 意为可缩放矢量图形
d2l.use_svg_display()

# 构建model
checkpoint = 'bert-large-uncased'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)  # 分词器
config = BertConfig.from_pretrained(checkpoint)

class BERTModel(nn.Module):
    def __init__(self, checkpoint, config):
        super(BERTModel, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained(checkpoint, config=config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        logit = outputs[0]
        return logit

# 构建数据集
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, sentences, attention_mask, token_type_ids, label):
        super(MyDataSet, self).__init__()
        self.sentences = torch.tensor(sentences, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        self.label = torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return self.sentences.shape[0]

    def __getitem__(self, idx):
        return self.sentences[idx], self.attention_mask[idx], self.token_type_ids[idx], self.label[idx]

# 加载数据，这里其实可以用DataLoader
def load_data(file_path):
    data = pd.read_csv(file_path, sep="\t", header=None, names=['sn', 'polarity', 'text'])
    data = data[data['polarity'] != 'neutral']  # 剔除neutral类型数据
    yy = data['polarity'].replace({'negative':0, 'positive':1, 'neutral':2})
    return data.values[:, 2:3].tolist(), yy.tolist()

pos_id = tokenizer.convert_tokens_to_ids('good')
neg_id = tokenizer.conveert_tokens_to_ids('bad')

# 数据预处理
mask_pos = 3
prefix = "It was [MASK]. "
def preprocess_data(file_path):
    x_train, y_train = load_data(file_path)
    Inputid = []
    Labelid = []
    token_type_ids = []
    attention_mask = []

    for i in range(len(x_train)):
        text = prefix + x_train[i][0]
        encode_dict = tokenizer.encode_plus(text, max_length=60, padding='max_length', truncation=True)
        input_ids = encode_dict['input_ids']
        token_type_ids.append(encode_dict['token_type_ids'])
        attention_mask.append(encode_dict['attention_mask'])
        label_id, input_id = input_ids[:], input_ids[:]
        if y_train[i] == 0:
            label_id[mask_pos] = neg_id
            label_id[:mask_pos] = [-1] * len(label_id[:mask_pos])
            label_id[mask_pos + 1:] = [-1] * len(label_id[mask_pos + 1:])
        else:
            label_id[mask_pos] = pos_id
            label_id[:mask_pos] = [-1] * len(label_id[:mask_pos])
            label_id[mask_pos + 1:] = [-1] * len(label_id[mask_pos + 1:])

        Labelid.append(label_id)
        Inputid.append(input_id)

    return Inputid, Labelid, token_type_ids, attention_mask

# 创建数据集
train_batch_size = 32
test_batch_size = 32
Inputid_train, Labelid_train, typeids_train, inputnmask_train = preprocess_data('/data/Twitter2013')
Inputid_dev, Labelid_dev, typeid_dev, inputnmask_dev = preprocess_data('/data/Twitter2013')
Inputid_test, Labelid_test, typeids_test, inputnmask_test = preprocess_data('/data/Twitter2013')

train_iter = DataLoader(MyDataSet(Inputid_train, inputnmask_train, typeids_train, Labelid_train), train_batch_size, True)
dev_iter = DataLoader(MyDataSet(Inputid_dev, inputnmask_dev, typeid_dev, Labelid_dev), train_batch_size, True)
test_iter = DataLoader(MyDataSet(Inputid_test, inputnmask_test, typeids_test, Labelid_test), test_batch_size, True)

train_len = len(Inputid_train)
test_len = len(Inputid_test)

train_loss = []
eval_loss = []

train_acc = []
eval_acc = []

# 训练函数
def train(net, train_iter, test_iter, lr, weight_decay, num_epochs):
    total_time = 0
    net = nn.DataParallel(net.to(device))
    loss = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=len(train_iter), num_training_steps=num_epochs*len(train_iter))
    for epoch in range(num_epochs):
        start_of_epoch = time.time()
        cor = 0
        loss_sum = 0
        net.train()
        for idx, (ids, att_mask, type, y) in enumerate(train_iter):
            optimizer.zero_grad()
            ids, att_mask, type, y = ids.to(device), att_mask.to(device), type.to(device), y.to(device)
            out_train = net(ids, att_mask, type)
            l = loss(out_train.view(-1, tokenizer.vocab_size), y.view(-1))
            l.backward()
            optimizer.step()
            schedule.step()
            loss_sum += l.item()
            if (idx + 1) % 20 == 0:
                print("Epoch {:04d} | Step {:06d} | Loss {:.4f} | Time {:.0f}".format(
                    epoch+1, idx+1, len(train_iter), loss_sum/(idx+1), time.time() - start_of_epoch
                ))
            truelabel = y[:, mask_pos]
            out_train_mask = out_train[:, mask_pos, :]
            predicted = torch.max(out_train_mask, 1)[1]
            cor += (predicted == truelabel).sum()
            cor = float(cor)

        acc = float(cor / train_len)

        eval_loss_sum = 0.0
        net.eval()
        correct_test = 0
        with torch.no_grad():
            for ids, att, tpe, y in test_iter:
                ids, att, tpe, y = ids.to(device), att.to(device), tpe.to(device), y.to(device)
                out_test = net(ids, att, tpe)
                loss_eval = loss(out_test.view(-1, tokenizer.vocab_size), y.view(-1))
                eval_loss_sum += loss_eval.item()
                ttruelabel = y[:, mask_pos]
                tout_train_mask = out_test[:, mask_pos, :]
                predicted_test = torch.max(tout_train_mask, 1)[1]
                correct_test += (predicted_test == truelabel).sum()
                correct_test = float(correct_test)
        acc_test = float(correct_test / test_len)

        if epoch % 1 == 0:
            print(('epoch {}, train_loss {}, train_acc {}, eval_loss {}, acc_test {}'.format(
                epoch+1, loss_sum/(len(train_iter)), acc, eval_loss_sum / (len(test_iter)), acc_test
            )))
            train_loss.append(loss / len(train_iter))
            eval_loss.append(eval_loss_sum / len(test_iter))
            train_acc.append(acc)
            eval_acc.append(acc_test)

        end_of_epoch = time.time()
        print('epoch {} duration:'.format(epoch+1), end_of_epoch - start_of_epoch)
        total_time += end_of_epoch - start_of_epoch
    print('total training time: ', total_time)

# start training
net = BERTModel(checkpoint, config)
num_epochs, lr, weight_decay = 20, 2e-5, 1e-4
print('baseline: ', checkpoint)
print('training...')
train(net, train_iter, test_iter, lr, weight_decay, num_epochs)

# 绘制acc/loss曲线
epoch = []
for i in range(num_epochs):
    epoch.append(i)

plt.figure()
plt.plot(epoch, train_acc, label='Train acc')
plt.plot(epoch, eval_acc, label='Test acc')
plt.title('Training and Testing accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylable('acc')
plt.figure()
plt.plot(epoch, train_loss, label='Train loss')
plt.plot(epoch, eval_loss, label='Test loss')
plt.title('Training and Testing loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('loss')

plt.show()














