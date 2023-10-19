import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict as edict

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import faces_data, High_Data, Low_Data
from model import High2Low


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
    data = faces_data(High_Data, Low_Data)
    test_loader = DataLoader(dataset=data, batch_size=opt.batchsize)
    
    net_G_h2l = High2Low().to(device)
    checkpoint = torch.load("/home/barc/Desktop/subir/Projects/Diffusion-main/HIgh2Low_GAN/intermid_results/models/model_epoch_001.pth")
    net_G_h2l.load_state_dict(checkpoint['G_h2l'])    
    net_G_h2l = net_G_h2l.eval()
    
    index = 0
    save_path = 'h2l_res'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    c = 0
    
    #print('======================== EVALUATE ============================')        
    for idx, data_dict in enumerate(test_loader):
        c += 1
        #print('======================== INSIDE ============================')
        print(idx)
        index = index + 1
        data_high = data_dict['hr']
        data_high_down = data_dict['hr_down']
        zs = data_dict['z']
        
        with torch.no_grad():
            data_input_high, _ = to_var(data_high)
            zs_cuda, _ = to_var(zs)
            data_new_low_output = net_G_h2l(data_input_high, zs_cuda)
            
        np_high = data_high.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_high = (np_high - np_high.min()) / (np_high.max() - np_high.min())
        np_high = (np_high * 255).astype(np.uint8)

        np_high_down = data_high_down.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_high_down = (np_high_down - np_high_down.min()) / (np_high_down.max() - np_high_down.min())
        np_high_down = (np_high_down * 255).astype(np.uint8)

        np_low_result = data_new_low_output.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
        np_low_result = (np_low_result - np_low_result.min()) / (np_low_result.max() - np_low_result.min())
        np_low_result = (np_low_result * 255).astype(np.uint8)
        
        cv2.imwrite("{}/{}_hr.png".format(save_path, idx), np_high) 
        cv2.imwrite("{}/{}_hr-down.png".format(save_path, idx), np_high_down)
        cv2.imwrite("{}/{}_lr-gen.png".format(save_path, idx), np_low_result)
        
        if c == 100:
            print(c)
            break
		
if __name__ == '__main__':
    main()
