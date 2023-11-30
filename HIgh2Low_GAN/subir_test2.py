import cv2
import os
import random
import pickle
import _pickle as cPickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import faces_data, High_Data, Low_Data, get_loader
from model import High2Low, Discriminator, GEN_DEEP


#parser = argparse.ArgumentParser()
#parser.add_argument("-c", "--gpu", action="store", dest="gpu", help="separate numbers with commas, eg. 3,4,5", required=True)


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
        
which_device = get_default_device()
#print(which_device)

# dbfile = open("model.pkl",'rb')
# db = pickle.load(dbfile, *, fix_import=True, )
# # for keys in db:
# #     print(keys, '=>', db[keys])
# print(dbfile)
# dbfile.close()


# with open(r"model.pkl", "rb") as dbfile:
#     db = cPickle.load(dbfile)

# print(db)  
# print(dbfile)

a = torch.load('model.pkl')
net_G_low2high.load_state_dict(a)