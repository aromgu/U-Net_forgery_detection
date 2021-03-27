import torch
from dataloader_lib import CreateDataset, split
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import cv2

train, test = split()

train = DataLoader(train, batch_size=4)
test = DataLoader(test, batch_size=4)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

PATH = '../model/model98.pt'

model = torch.load(PATH)
criterion = nn.CrossEntropyLoss()
epochs = 1
class_num = 2
count = 0
with torch.no_grad():
    model.eval()
    eval_loss = []
    for x, t in tqdm(test):
        for i,j in zip(x,t):
            i = i.to(device)
            i= torch.unsqueeze(i,0)
            j = j.to(device).long()
            j=torch.unsqueeze(j,0)
            target = j.squeeze()
            prediction = model(i)  # shape = [batch_size, 2]
            prediction1 = prediction.permute(0,2,3,1).reshape(-1, class_num)
            j = j.reshape(-1)
            loss = criterion(prediction1, j)
            eval_loss += [loss.item()]

            plot_img = prediction.squeeze()
            plot_img = torch.argmax(plot_img, dim=0)
            i = i.permute(2,3,1,0)
            i = i.squeeze()

            fig, ax = plt.subplots(1,3)
            ax[0].imshow(i.cpu())
            ax[1].imshow(plot_img.cpu(), cmap='gray')
            ax[2].imshow(target.cpu(), cmap='gray')
            ax[0].set_title('eval loss : %0.5f' % (np.mean(eval_loss)))
            ax[1].set_title('Predict')
            ax[2].set_title('GT')
            plt.savefig('./result/result%s.png'%(count))
            count += 1

            # plt.show()

    # print(np.mean(eval_loss))

# for epoch in range(epochs):
#     train_loss = train()
#     eval_loss = evaluation()