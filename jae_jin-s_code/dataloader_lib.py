from torch.utils.data import Dataset, DataLoader
from PIL import Image
import cv2
import os
import torch
from torchvision import transforms
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import  os.path


class CreateDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        # self.img_paths = sorted([img_path + '/' + x for x in os.listdir(img_path) if x.endswith('bmp')])
        # self.mask_paths = sorted([mask_path + '/' + x for x in os.listdir(mask_path) if x.endswith('bmp')])
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.n_classes = 2
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # print(self.img_paths)
        image = Image.open(self.img_paths[idx])
        label = Image.open(self.mask_paths[idx])

        image = image.convert('RGB')
        label = label.convert('L')

        X = self.transform(image)
        lab = self.transform(label)

        return X, lab

root_dir = '../../Datasets/CVIP/img'

def split():
    # img_paths = sorted([img_path + '/' + x for x in os.listdir(img_path) if x.endswith('bmp')])
    # mask_paths = sorted([mask_path + '/' + x for x in os.listdir(mask_path) if x.endswith('bmp')])
    # img_paths = [x for x in os.path.join(img_path) if x.endswith('bmp')]
    # mask_paths = [x for x in glob.glob(os.path.join(img_paths,'_mask')) if x.endswith('bmp')]
    for root, dirs, files in os.walk('../../Datasets/CVIP/img'):
        # img_paths = [x for x in os.listdir(root_dir+img_path+files) if x.endswith('bmp')]
        # mask_paths = [x for x in os.listdir(root_dir+mask_path+files+'_mask') if x.endswith('bmp')]
        img_paths = []
        mask_paths = []
        try:
            for name in files:
                img_pathss = os.path.join(root,name[:-3]+'bmp')
                mask_pathss = os.path.join('../../Datasets/CVIP/mask',name[:-4]+'_mask'+'.bmp')
                img_paths.append(img_pathss)
                mask_paths.append(mask_pathss)
                # print(img_pathss)
        except:
            pass

    x_train, x_test, y_train, y_test = train_test_split(img_paths, mask_paths, test_size=0.2, shuffle=False)
    return CreateDataset(x_train, y_train), CreateDataset(x_test, y_test)

if __name__ == '__main__':
    train, test = split()

    train = DataLoader(train, batch_size=4)
    test = DataLoader(test, batch_size=4)
    for x,t in train:
        for xx, tt in zip(x,t):
            fig, ax = plt.subplots(1,2)
            xx = np.transpose(xx, (1,2,0))
            tt = np.transpose(tt, (1,2,0))
            # ax[0].imshow(xx)
            # ax[1].imshow(tt, cmap='gray')
            # plt.show()

# prob = Image.open('../../Datasets/CVIP/mask/out_r5_im1_mask.bmp')
# aa = Image.open('../../Datasets/CVIP/mask/out_r5_im10_mask.bmp')
# prob.show()