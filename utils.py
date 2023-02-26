# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 20:04:46 2023

@author: lab503
"""
import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from itertools import product
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import tqdm
import torchvision.models as models
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from inc3 import InceptionV3
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model_inv = InceptionV3([block_idx])
model_inv = model_inv.to(device)

from dataset.mnistset import create_loader

from testing import calculate_activation_statistics
def get_mu_sigma(test_eval):
    real_images_all = []
    with torch.no_grad():
        
        for idx, (image, target) in enumerate(test_eval):
            image = image.to(device)
            
            image = image.expand(-1,3,-1,-1)
            target = target.to(device)
            
            
           
            real_images_all.append(image)
            
        real_images_all = torch.cat(real_images_all)
       
      
        real_images_all = real_images_all.to(device)
        mu1, sigma1 = calculate_activation_statistics(real_images_all,model_inv)
        return mu1, sigma1