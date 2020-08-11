import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras as tk

def get_means_and_stds(imgs): #[NHWC]
    means = np.mean(imgs,axis=(0,1,2))
    stds = np.std(imgs,axis=(0,1,2))
    return np.reshape(means,[1,1,1,-1]),\
            np.reshape(stds,[1,1,1,-1])

def normalize(imgs,means=0.0,stds=1.0):
    return (imgs-means)/stds

def denormalize(imgs,means=0.0,stds=1.0):
    return imgs*stds+means

def pad_or_crop(imgs,size,pad_mode="reflect"): # imgs:[NHWC] size:[w,h]
    padding = [(0,0),(0,0),(0,0),(0,0)]

    tmp = size[1] - imgs.shape[1]
    if tmp >= 0:
        padding[1] = (tmp//2,(tmp+1)//2)
    else:
        start_index,end_index = (-tmp)//2,-(-tmp+1)//2
        imgs = imgs[:,start_index:end_index,:,:]

    tmp = size[0] - imgs.shape[2]
    if tmp >= 0:
        padding[2] = (tmp//2,(tmp+1)//2)
    else:
        start_index,end_index = (-tmp)//2,-(-tmp+1)//2
        imgs = imgs[:,:,start_index:end_index,:]

    imgs = np.pad(imgs,padding,pad_mode)
    return imgs

if __name__ == "__main__":
    imgs = np.random.randint(0,255,[3,321,321,3],np.uint8).astype(np.float32)

    tmp = normalize(imgs)
    denormalize(tmp)
    
    tmp = pad_or_crop(imgs,[322,320])
    print(tmp.shape)
