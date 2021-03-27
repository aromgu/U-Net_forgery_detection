from dataloader_lib import CreateDataset, split
from torch.utils.data import DataLoader
from models import U_Net
from torch import optim
from torch import nn
import torch
from tqdm import tqdm
import numpy as np
from mask_one_hot import get_one_hot_encoded_mask

train, test = split()

train = DataLoader(train, batch_size=4)
test = DataLoader(test, batch_size=4)
# testloader = DataLoader(Testdataset, batch_size=4, shuffle=False, drop_last=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = U_Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
epochs = 100
count = 0
pre_cost = float('inf')
class_num = 2

for epoch in range(epochs):
    train_loss = []
    model.train()
    for x, t in tqdm(train):
        x = x.to(device)
        t = t.to(device).long()
        # t = torch.reshape(t, (1, -1))

        # t = get_one_hot_encoded_mask(t)
        optimizer.zero_grad()
        prediction = model(x)#.squeeze(-1)  # shape = [batch_size, 1]

        prediction = prediction.permute(0,2,3,1).reshape(-1, class_num)
        t = t.reshape(-1)
        # prediction = torch.reshape(prediction, (3,16384))
        # t = torch.reshape(t, (3,16384))
        loss = criterion(prediction, t)
        train_loss += [loss.item()]

        loss.backward()
        optimizer.step()

    cur_cost = np.mean(train_loss)

    if cur_cost < pre_cost:
        torch.save(model, f'../model/model{epoch}.pt')
        pre_cost = cur_cost
        count = 0
    else:
        count += 1

    pprint = f'epoch = {epoch}, pre_cost = {pre_cost}, cur_cost{cur_cost}, count = {count}\n'
    print(pprint)
    print('train loss: ',np.mean(train_loss))


    # with torch.no_grad():
    #     model.eval()
    #     eval_loss = []
    #     for x, t in tqdm(testloader):
    #         x = x.to(device)
    #         t = t.to(device)
    #         prediction = model(x)  # shape = [batch_size, 2]
    #         loss = criterion(prediction, t)
    #         eval_loss += [loss.item()]
    #
    #     print(np.mean(eval_loss))

# for epoch in range(epochs):
#     train_loss = train()
#     eval_loss = evaluation()