import cv2
import os
import numpy as np
from glob import glob, iglob

import PIL
from PIL import Image

import torch
import torch.nn.functional as nnF
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


High_Data = ["/home/barc/Desktop/subir/datasets/MS1MV2-50K/data"] #"D:/Face/Small-CelebA/images"]
Low_Data = ["/home/barc/Desktop/subir/datasets/Widerface/cropped_faces/cropped_images_train"] #"D:/Face/Wider_Face/cropped_images/cropped_images_train"]



class faces_super(Dataset):
    def __init__(self, datasets, transform):
        assert datasets, print('no datasets specified')
        self.transform = transform
        self.img_list = []
        dataset = datasets
        if dataset == 'widerfacetest':
            img_path = '/home/barc/Desktop/subir/datasets/Widerface/cropped_faces/cropped_images_val' #'D:/Face/Wider_Face/cropped_images/cropped_images_val/'
            list_name = (glob(os.path.join(img_path, "*.png")))
            list_name.sort()
            for filename in list_name:#jpg
                self.img_list.append(filename)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp16 = Image.open(self.img_list[index])
        inp64 = inp16.resize((64, 64), resample=PIL.Image.BICUBIC)
        data['img64'] = self.transform(inp64)
        data['img16'] = self.transform(inp16)
        data['imgpath'] = self.img_list[index]
        return data

def get_loader(dataname, bs=1):
    transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    dataset = faces_super(dataname, transform)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader
    

class faces_data(Dataset):
    def __init__(self, data_hr, data_lr):
        self.hr_imgs = [os.path.join(d, i) for d in data_hr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_imgs = [os.path.join(d, i) for d in data_lr for i in os.listdir(d) if os.path.isfile(os.path.join(d, i))]
        self.lr_len = len(self.lr_imgs)
        self.lr_shuf = np.arange(self.lr_len)
        np.random.shuffle(self.lr_shuf)
        self.lr_idx = 0
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        #print(len(self.lr_imgs))
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index])
        hr = cv2.resize(hr, (64, 64), interpolation = cv2.INTER_CUBIC)
        lr = cv2.imread(self.lr_imgs[self.lr_shuf[self.lr_idx]])
        lr = cv2.resize(lr, (16, 16), interpolation = cv2.INTER_CUBIC)
        self.lr_idx += 1
        if self.lr_idx >= self.lr_len:
            self.lr_idx = 0
            np.random.shuffle(self.lr_shuf)
        data["z"] = torch.randn(1, 64, dtype=torch.float32)
        data["lr"] = self.preproc(lr)
        data["hr"] = self.preproc(hr)
        data["hr_down"] = nnF.avg_pool2d(data["hr"], 4, 4)
        return data
    
    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)

if __name__ == "__main__":
    test_loader = get_loader("widerfacetest", bs=1)
    for i, sample in enumerate(test_loader):
        if i >= 3: 
            break
        low_temp = sample["img16"].numpy()
        low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :]))
        np_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
        np_low = (np_low * 255).astype(np.uint8)
        cv2.imshow("Low", np_low)
        cv2.waitKey()
        cv2.destroyAllWindows()
        
    # data = faces_data(High_Data, Low_Data)
    # loader = DataLoader(dataset=data, batch_size=16, shuffle=True)
    # for i, batch in enumerate(loader):
    #     print("batch: ", i)
    #     lrs = batch["lr"].numpy()
    #     hrs = batch["hr"].numpy()
    #     downs = batch["hr_down"].numpy()

    #     for b in range(batch["z"].size(0)):
    #         lr = lrs[b]
    #         hr = hrs[b]
    #         down = downs[b]
    #         lr = lr.transpose(1, 2, 0)
    #         hr = hr.transpose(1, 2, 0)
    #         down = down.transpose(1, 2, 0)
    #         lr = (lr - lr.min()) / (lr.max() - lr.min())
    #         hr = (hr - hr.min()) / (hr.max() - hr.min())
    #         down = (down - down.min()) / (down.max() - down.min())
    #         cv2.imshow("lr-{}".format(b), lr)
    #         cv2.imshow("hr-{}".format(b), hr)
    #         cv2.imshow("down-{}".format(b), down)
    #         cv2.waitKey()
    #         cv2.destroyAllWindows()

    print("finished.")
