import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

import torch
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from dataset import get_loader
from model import GEN_DEEP


os.sys.path.append(os.getcwd())

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
which_device = get_default_device()
print(which_device)

def to_var(data):
    device = "cuda"
    real_cpu = data
    batchsize = real_cpu.size(0)
    input = Variable(real_cpu.to(device))
    return input, batchsize

def main():
    device = "cuda"
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    opt = edict()
    opt.nGPU = 1
    opt.batchsize = 1
    opt.cuda = True
    cudnn.benchmark = True
    
    print('========================LOAD DATA============================')
    data_name = 'widerfacetest'
    test_loader = get_loader(data_name, opt.batchsize)
    
    net_G_low2high = GEN_DEEP()
    net_G_low2high = net_G_low2high.to(device)

    checkpoint = torch.load("/home/barc/Desktop/subir/Projects/Diffusion-main/HIgh2Low_GAN/intermid_results/models/model_epoch_050.pth")
    net_G_low2high.load_state_dict(checkpoint['G_l2h'])    
    net_G_low2high = net_G_low2high.eval()
    index = 0
    test_file = 'test_res'
    if not os.path.exists(test_file):
        os.makedirs(test_file)

    #print('======================== EVALUATE ============================')        
    for idx, data_dict in enumerate(test_loader):
        #print('======================== INSIDE ============================')
        print(idx)
        index = index + 1
        data_low = data_dict['img16']
        data_high = data_dict['img64']
        #print(data_dict['imgpath'])
        img_name = data_dict['imgpath'][0].split('/')[-1] #Fetch just the image name
        #print(img_name)
        with torch.no_grad():
            data_input_low, batchsize_high = to_var(data_low)
            data_input_high, _ = to_var(data_high)
            data_high_output = net_G_low2high(data_input_low)
            
        np_low = data_low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
        np_low = (np_low * 255).astype(np.uint8)

        np_high = data_high.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_high = (np_high - np_high.min()) / (np_high.max() - np_high.min())
        np_high = (np_high * 255).astype(np.uint8)

        np_result = data_high_output.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_result = (np_result - np_result.min()) / (np_result.max() - np_result.min())
        np_result = (np_result * 255).astype(np.uint8)
        
        cv2.imwrite("test_res/{}_low.png".format(idx), np_low)
        cv2.imwrite("test_res/{}_high.png".format(idx), np_high)
        cv2.imwrite("test_res/{}_sr.png".format(idx), np_result)
        
        #cv2.imshow("low", np_low[:, :, ::-1])
        #cv2.imshow("high", np_high[:, :, ::-1])
        #cv2.imshow("result", np_result[:, :, ::-1])
        #cv2.waitKey()
        
        #plt.imshow(np_low[:, :, ::-1])
        #plt.imshow(np_high[:, :, ::-1])
        #plt.imshow(np_result[:, :, ::-1])
        
        # path = os.path.join(test_file, img_name.split('.')[0]+'.jpg')
        # vutils.save_image(data_high_output.data, path, normalize=True)

if __name__ == '__main__':
    main()
