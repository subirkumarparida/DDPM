import os
import cv2
import numpy as np
import pandas as  pd
from matplotlib import pyplot as plt

#i = 10

save_name = 'output.csv'
df = pd.read_csv(save_name, index_col=0)

maxValues = df.max(axis=1)
maxValueIndex = df.idxmax(axis=1)

def display_image(gen_img, train_img, f_name):
    
    fig = plt.figure(figsize=(10,8))

    gen_img = cv2.imread(gen_img)
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    gen_img = gen_img[:, :, ::-1]    
    
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(gen_img)
    ax1.set_title("Diffusion")
    ax1.axis('off')

    
    train_img = cv2.imread(train_img)
    # Convert the image from BGR color (which OpenCV uses) to RGB color
    train_img = train_img[:, :, ::-1]    
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(train_img)
    ax2.set_title("Dataset")
    ax2.axis('off')

    fig.savefig(f_name)

for i in np.arange(20):
    file_name = 'output/i_' + str(i) + '.jpg'
    df1 = df.loc[df[maxValueIndex[i]] == maxValues[i]]
    display_image(df1.index[0], maxValueIndex[i], file_name)