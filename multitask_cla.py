import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
import torch
from torch import nn
from transformers import BertJapaneseTokenizer,BertConfig,BertModel,AdamW


max_length = 300
BATCH_SIZE = 2
input_ids, input_masks = [], []
scores1,scores2,scores3,scores4 = [],[],[],[]

bert_path = 'cl-tohoku/bert-base-japanese-whole-word-masking'
tokenizer = BertJapaneseTokenizer.from_pretrained(bert_path)


with open("data_4.csv", encoding='utf-8') as f:
  for i, line in enumerate(f):
        text,score1,score2,score3,score4 = line.strip().split(',')
        encode_dict = tokenizer.encode_plus(text=text,max_length=max_length,
                                            padding='max_length', truncation=True)
        input_ids.append(encode_dict['input_ids'])
        input_masks.append(encode_dict['attention_mask'])
        scores1.append(int(float(score1)))
        scores2.append(int(float(score2)))
        scores3.append(int(float(score3)))
        scores4.append(int(float(score4)))


input_ids, input_masks = np.array(input_ids), np.array(input_masks)
scores1 = np.array(scores1)
scores2 = np.array(scores2)
scores3 = np.array(scores3)
scores4 = np.array(scores4)

print(input_ids.shape, input_masks.shape, scores1.shape)
print(input_ids[0],input_masks[0],scores1[0],scores2[0],scores3[0],scores4[0])

print(input_ids.shape,input_masks.shape,scores1.shape)

idxes = np.arange(input_ids.shape[0])
np.random.seed(2021)   # 固定
np.random.shuffle(idxes)
print(idxes.shape, idxes[:10])

input_ids_train, input_ids_test = input_ids[idxes[:300]], input_ids[idxes[300:]]
input_masks_train, input_masks_test = input_masks[idxes[:300]],input_masks[idxes[300:]]
y_train1, y_test1 = scores1[idxes[:300]], scores1[idxes[300:]]
y_train2, y_test2 = scores2[idxes[:300]], scores2[idxes[300:]]
y_train3, y_test3 = scores3[idxes[:300]], scores3[idxes[300:]]
y_train4, y_test4 = scores4[idxes[:300]], scores4[idxes[300:]]

print(y_train3)

train_data = TensorDataset(torch.LongTensor(input_ids_train),
                           torch.LongTensor(input_masks_train),
                           torch.LongTensor(y_train1),
                           torch.LongTensor(y_train2),
                           torch.LongTensor(y_train3),
                           torch.LongTensor(y_train4))
train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
print(train_loader)
# for idx, (ids, att, y1,y2,y3,y4) in enumerate(train_loader):
#     print(idx,ids,att,y1,y2,y3,y4)
#     break

test_data = TensorDataset(torch.LongTensor(input_ids_test),
                          torch.LongTensor(input_masks_test),
                          torch.LongTensor(y_test1),
                          torch.LongTensor(y_test2),
                          torch.LongTensor(y_test3),
                          torch.LongTensor(y_test4))

test_loader = DataLoader(test_data, shuffle=True, batch_size=BATCH_SIZE)
for idx, (ids, att, y1,y2,y3,y4) in enumerate(test_loader):
    print(idx,ids,att,y1,y2,y3,y4)
    break

class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=6):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)
        self.config.output_hidden_states = True
        self.bert = BertModel.from_pretrained(bert_path,config = self.config)
        self.fc1 = nn.Linear(self.config.hidden_size*max_length, classes)
        self.fc2 = nn.Linear(self.config.hidden_size * max_length, classes)
        self.fc3 = nn.Linear(self.config.hidden_size * max_length, classes)
        self.fc4 = nn.Linear(self.config.hidden_size * max_length, classes)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask)
        #output[0]: torch.Size([batch_size, 300, 768])最后の層　outputs[1]はels:torch.Size([batch_size, 768])　output[2]は１３層
        out1 = outputs[2][11]  #len(outputs[2])) = 13   torch.Size([batch_size, 300, 768]) 最后第２層
        pre1 = self.fc1(out1.view(-1,self.config.hidden_size * max_length))

        out2 = outputs[2][10]
        pre2 = self.fc2(out2.view(-1,self.config.hidden_size * max_length))
        out3 = outputs[2][9]
        pre3 = self.fc3(out3.view(-1,self.config.hidden_size * max_length))
        out4 = outputs[2][8]
        pre4 = self.fc4(out4.view(-1,self.config.hidden_size * max_length))
        return pre1,pre2,pre3,pre4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 5
model = Bert_Model(bert_path).to(DEVICE)
# if os.path.exists("./model4.pkl"):
#    model = torch.load("./model4.pkl")
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    for i in range(epoch):
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        for idx, (ids, att, y1,y2,y3,y4) in enumerate(train_loader):
            ids, att, y1,y2,y3,y4 = ids.to(DEVICE), att.to(DEVICE), y1.to(DEVICE),y2.to(DEVICE),y3.to(DEVICE),y4.to(DEVICE)
            y_pred1,y_pred2,y_pred3,y_pred4 = model(ids, att)
            loss1 = criterion(y_pred1, y1)
            loss2 = criterion(y_pred2, y2)
            loss3 = criterion(y_pred3, y3)
            loss4 = criterion(y_pred4, y4)
            optimizer.zero_grad()
            total_loss = (loss1 + loss2 + loss3 + loss4) / 4
            total_loss.backward()
            optimizer.step()
            if (idx + 1) % 5 == 0:
                print("Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f}({:.4f},{:.4f},{:.4f},{:.4f})".format(
                    i + 1, idx + 1, len(train_loader), total_loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()))
        acc = evaluate(model)
        print("Test DataSet ACC = {}".format(acc))
    #torch.save(model, "./model4.pkl")  #保存整个模型

def evaluate(model):
    model.eval()
    val_pred1,val_pred2,val_pred3,val_pred4,val_true1,val_true2,val_true3,val_true4 = [],[],[],[],[],[],[],[]
    with torch.no_grad():
        for idx, (ids, att, y1,y2,y3,y4) in (enumerate(test_loader)):
            y_pred1,y_pred2,y_pred3,y_pred4 = model(ids.to(DEVICE), att.to(DEVICE))
            y_pred1 = torch.argmax(y_pred1, dim=1).detach().cpu().numpy().tolist()
            val_pred1.extend(y_pred1)
            val_true1.extend(y1.numpy().tolist())

    return accuracy_score(val_true1, val_pred1)  # 返回accuracy

acc = evaluate(model)
print("Test DataSet ACC {}".format(acc))
train(5)