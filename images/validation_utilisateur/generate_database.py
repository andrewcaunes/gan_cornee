#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 11:02:21 2021

@author: yann
"""

import glob
import numpy as np
import skimage.io

FOLDER_REAL_IMAGES="../curated_dataset/train/*.png"
FOLDER_SIMU_IMAGES="../../results/{}.png"
FOLDER_OUTPUT="data_validation/"

n_simu = 1024

names_simu=[]
# nom des fichiers simulés
for i in range(n_simu):
    names_simu.append(FOLDER_SIMU_IMAGES.format(i))
    
# nom des fichiers réels
names_real =  glob.glob(FOLDER_REAL_IMAGES)

# Nombre d'images à générer dans chaque catégorie
N = 1000

# Mélange
melange = np.zeros(2*N)
melange[N:]=1
np.random.shuffle(melange)


def extractImage(I, t=128):
    if I.ndim == 3:
        I = I[:,:,1]
    s = I.shape 
    x = s[0]-t
    y = s[1]-t
    x0 = np.random.randint(x)
    y0 = np.random.randint(y)
    
    crop = I[x0:x0+t, y0:y0+t]
    return crop


n_simu = len(names_simu)
n_real = len(names_real)
output_names=[]

for i, label in enumerate(melange):
    if label:
        index = np.random.randint(n_real)
        name = names_real[index]
    else:
        index = np.random.randint(n_simu)
        name = names_simu[index]
    
    I = skimage.io.imread(name)
    if label:
        crop = extractImage(I)
    else:
        crop = I
    outputfile = "validation_{:04d}.png".format(i)
    skimage.io.imsave(FOLDER_OUTPUT+outputfile, crop)
    output_names.append(outputfile)
    
with open(FOLDER_OUTPUT+'labels.npy', 'wb') as f:
    np.save(f, np.array(output_names))
    np.save(f, melange)
    

with open(FOLDER_OUTPUT+'labels.npy', 'rb') as f:
    a = np.load(f)
    b = np.load(f)
    
print(a, b)