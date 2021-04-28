#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 13:20:23 2021
Calcul de la performance de la classification manuelle

@author: yann
"""

import numpy as np
import glob
import os

REP = "test_hasard2/"

reel = glob.glob(REP+"reel/*.png")

FOLDER_OUTPUT="data_validation/"
    
with open(FOLDER_OUTPUT+'labels.npy', 'rb') as f:
    names = np.load(f)
    labels = np.load(f)
    
classification = np.zeros(labels.shape)
for indice, name in enumerate(names):
    if os.path.exists(REP+"reel/" + name):
        classification[indice] = 1
    else:
        if os.path.exists(REP+"simulation/" + name):
            classification[indice] = 0
        else:
            print("ERREUR: fichier non trouvé: ", name)
        
        

# plot confusion matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_cm(cm, classes, normalize=False, cmap=plt.cm.Blues):
    """
    Plot confusion matrix
    cm: confusion matrix, as ouput by sklearn.metrics.confusion_matrix
    classes: labels to be used
    normalize: display number (False by default) or fraction (True)
    cmap: colormap

    returns: figure that can be used for pdf export
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fmt = '.2f' if normalize else 'd'
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)

    fig = plt.figure()
    sn.set(font_scale=.8)
    sn.heatmap(df_cm, annot=True, cmap=cmap, fmt=fmt)
    plt.xlabel('Classification label')
    plt.ylabel('True label')
    plt.show()

    return fig

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
target_names=['image simulée', 'image réelle']
print(classification_report(labels, classification, target_names=target_names))
fig = plot_cm(confusion_matrix(labels, classification), target_names, True)
