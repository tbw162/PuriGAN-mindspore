# -*- coding: utf-8 -*-


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
import copy
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch
import inspect
import mindspore
import mindspore.ops as ops
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torchvision.models as models
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    
    batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)


    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)
    startt = time.time()
    returnvalue = (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)
    endt = time.time()
    
    return returnvalue

def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value
 
def testing(model_inv,mu1,sigma1,generator,batches_done,opt):
    generator.eval()
 
    eva_dic = {}
    
    
    fake_images_all = []
    for i in range(0,10):
        z_out_fake = ops.StandardNormal()((100, opt.latent_dim, 1, 1))
   
        gen = generator(z_out_fake)
        if(opt.dataset=='MNIST' or opt.dataset == 'F-MNIST'):
            gen = gen.expand(-1,3,-1,-1)
        
      
    
        fake_images_all.append(gen.asnumpy())
        
    
    fake_images_all = np.concatenate(fake_images_all,axis=0)
    fake_images_all = torch.from_numpy(fake_images_all)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fake_images_all = fake_images_all.to(device)
    mu2, sigma2 = calculate_activation_statistics(fake_images_all,model_inv)
    fid_value = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    eva_dic['fid'] = fid_value
    return eva_dic

