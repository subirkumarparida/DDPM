import os
import cv2
import numpy as np
import pandas as  pd
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary
from model.resnet import Resnet34Triplet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

train_imgs_path = '../DDGAN/data/celebahq256_imgs/train/train/'
#train_imgs_path = '../DDGAN/data/celebahq256_imgs/valid/valid/'
gen_imgs_path = '../FairFace-master/celeba_hq_gen_ddgan/'

checkpoint = torch.load('model/model_resnet34_triplet.pt', map_location=device)
model = Resnet34Triplet(embedding_dimension=checkpoint['embedding_dimension'])
model.load_state_dict(checkpoint['model_state_dict'])
best_distance_threshold = checkpoint['best_distance_threshold']

model.to(device)
model.eval()

#summary(model, input_size = (3, 112, 112), batch_size = -1)

similarity = nn.CosineSimilarity(dim=0, eps=1e-6)

def create_embedding(imgs_path, save_name, face_model):
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(140,140)),  # Pre-trained model uses 140x140 input images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.6071, 0.4609, 0.3944],  # Normalization settings for the model, the calculated mean and std values
                    std=[0.2457, 0.2175, 0.2129]     # for the RGB channels of the tightly-cropped glint360k face dataset
                )
    ])
    
    img_names = [os.path.join(imgs_path, x) for x in os.listdir(imgs_path)]
    face_names = []
    embeddings_list = []
    
    for index, img_name in enumerate(img_names):
        if index % 1000 == 0:
            print("Computing embedding... {}/{}".format(index, len(img_names)))
        
        #print(img_name)
        face_names.append(img_name)
        #print(face_names)
        
        img = cv2.imread(img_name)  # Or from a cv2 video capture stream

        # Note that you need to use a face detection model here to crop the face from the image and then
        #  create a new face image object that will be inputted to the facial recognition model later.

        # Convert the image from BGR color (which OpenCV uses) to RGB color
        img = img[:, :, ::-1]
    
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(device)

        embedding = face_model(img)
        embedding = torch.flatten(embedding)
        embedding = embedding.cpu().detach().numpy()
        
#         embedding = embedding.cpu().detach().numpy()
#         embedding = np.squeeze(embedding)
        embeddings_list.append(embedding)
#         result = pd.DataFrame()
#         px = pd.DataFrame(embedding)
#         result = pd.concat([result, px], axis=1, ignore_index=True)

    result = pd.DataFrame(embeddings_list).T
    result.columns = img_names
    result.to_csv(save_name)
    #print(img_names)
    
    return result

train_csv = 'train_embeddings.csv'
if not(os.path.isfile(train_csv)):
    df_train = create_embedding(train_imgs_path, train_csv, model)

gen_csv = 'gen_embeddings.csv'
if not(os.path.isfile(gen_csv)):
    df_gen = create_embedding(gen_imgs_path, gen_csv, model)


def compute_all_scores(gen_imgs_path, train_imgs_path, gen_csv, train_csv, save_name, sim_metric=similarity):
      
    df_gen = pd.read_csv(gen_csv, index_col=0)
    gen_img_names = [os.path.join(gen_imgs_path, x) for x in os.listdir(gen_imgs_path)]
    gen_face_names = []
    
    df_train = pd.read_csv(train_csv, index_col=0)
    train_img_names = [os.path.join(train_imgs_path, x) for x in os.listdir(train_imgs_path)]
    train_face_names = []
#     print(df_train)
    
    g_score_list = []
    
    for g_index, gen_img_name in enumerate(gen_img_names):
        if g_index % 100 == 0:
                print("Gen data scores ... {}/{}".format(g_index, len(gen_img_names)))
        
        gen_img_embed = df_gen[gen_img_name]
        gen_img_embed = torch.Tensor(gen_img_embed)
        
        t_score_list = []
        
        for t_index, train_img_name in enumerate(train_img_names):
            # if t_index % 10000 == 0:
            #     print("Training data scores ... {}/{}".format(t_index, len(train_img_names)))
            train_img_embed = df_train[train_img_name]
            train_img_embed = torch.Tensor(train_img_embed)
            
            score = sim_metric(gen_img_embed, train_img_embed)
            #print(score)
            t_score_list.append(score.item())
        #print(t_score_list)
        
        g_score_list.append(t_score_list)
    #print(g_score_list)
    
    result = pd.DataFrame(g_score_list)
    result.columns = train_img_names
    result.index = gen_img_names
    result.to_csv(save_name)
    
    return result

save_name = 'output.csv'

df_score = compute_all_scores(gen_imgs_path, train_imgs_path, gen_csv, train_csv, save_name)

df = pd.read_csv(save_name, index_col=0)
#df

maxValues = df.max(axis=1)
maxValueIndex = df.idxmax(axis=1)
#print(maxValues)
#print(maxValueIndex)

# for i in np.arange(3):
#     #print(i)
#     df1 = df.loc[df[maxValueIndex[i]] == maxValues[i]]
#     print(df1.index[0], maxValues[i], maxValueIndex[i])


def pandas_rank_based(df, n_max):
    #n_max.index = df.index
    df_rank = df.stack(dropna=False).groupby(level=0).rank(ascending=False, method='first').unstack()
    selected = df_rank.le(n_max, axis=1)
    return df[selected]

sel_df = pandas_rank_based(df, 2)
#sel_df