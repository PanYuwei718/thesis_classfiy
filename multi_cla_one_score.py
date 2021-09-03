import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from transformers import BertJapaneseTokenizer,BertConfig,BertModel,AdamW
import os


max_length = 500
BATCH_SIZE = 4
input_ids, input_masks = [], []
scores = []

bert_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_path)


with open("data_3_1.csv", encoding='utf-8') as f:
  for i, line in enumerate(f):
        text, score = line.strip().split(',')

        encode_dict = tokenizer.encode_plus(text=text,max_length=max_length,
                                            padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_masks.append(encode_dict['attention_mask'])
        scores.append(int(float(score)))


input_ids, input_masks = np.array(input_ids), np.array(input_masks)
scores = np.array(scores)

print(input_ids.shape, input_masks.shape, scores.shape)
print(input_ids[0],input_masks[0],scores[0])

print(input_ids.shape,input_masks.shape,scores.shape)


idxes = np.arange(input_ids.shape[0])
np.random.seed(2021)   # 固定
np.random.shuffle(idxes)
print(idxes.shape, idxes[:10])

input_ids_train, input_ids_test = input_ids[idxes[:900]], input_ids[idxes[900:]]
input_masks_train, input_masks_test = input_masks[idxes[:900]],input_masks[idxes[900:]]
y_train, y_test = scores[idxes[:900]], scores[idxes[900:]]


train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train))
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test))
test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)


class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=6):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(bert_path,config = self.config)
        self.fc = nn.Linear(self.config.hidden_size * max_length, classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask)
        #output[0]: torch.Size([batch_size, 300, 768])最后の層　      outputs[1]:   els:torch.Size([batch_size, 768])
        out = outputs[2][11]  #len(outputs[2])) = 13   torch.Size([batch_size, 300, 768]) 最后第２層
        pre = self.fc(out.view(-1,self.config.hidden_size * max_length))
        return pre

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
model = Bert_Model(bert_path).to(DEVICE)

# if  os.path.exists("./model3_1.pkl"):
#     model= torch.load("./model3_1.pkl")
optimizer = AdamW(model.parameters(), lr=2e-5)

criterion = nn.CrossEntropyLoss()

def train(epoch):
    for i in range(epoch):
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        for idx, (ids, att, y) in enumerate(train_loader):
            ids, att, y = ids.to(DEVICE), att.to(DEVICE), y.to(DEVICE)
            y_pred = model(ids, att)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            if (idx + 1) % 10 == 0:  # 设定每个batch中的多少个step打印一次结果。
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f}".format(
                    i + 1, idx + 1, len(train_loader), train_loss_sum / (idx + 1)))
        acc = evaluate(model)
        print("Test DataSet ACC = {}".format(acc))
    # torch.save(model, "./model3_1.pkl")  #保存整个模型

def evaluate(model):
    model.eval()
    val_pred, val_true = [], []
    with torch.no_grad():
        for idx, (ids, att, y) in enumerate(test_loader):
            ids, att = ids.to(DEVICE), att.to(DEVICE)
            y_pred = model(ids,att)
            y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
            val_pred.extend(y_pred)
            val_true.extend(y.numpy().tolist())
    return accuracy_score(val_true, val_pred)  #返回accuracy

acc = evaluate(model)
print("Test DataSet ACC {}".format(acc))
train(3)