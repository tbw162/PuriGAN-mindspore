# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import math
from itertools import product

import tqdm
import copy
import mindspore as nn
import pandas as pd
import mindspore.ops as ops
import inspect
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from itertools import cycle
import warnings
import time
a = 1
b = 0 
PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--gpu", type=str, default='3', help="gpu_num")
parser.add_argument("--dir", type=str, default='/summary//', help="save dir")
parser.add_argument("--name", type=str, default='fidresult', help="file name")
parser.add_argument("--seed", type=int,default=13,help="random seed")
parser.add_argument("--dataset", type=str, default='F-MNIST', help="choice of dataset(F-MNIST,MNIST,CELEBA,SVHN)")
parser.add_argument("--gamma_p", type=float, default=0.5, help="ratio of pollution data")
parser.add_argument("--gamma_c", type=float, default=0.2, help="ratio of collected pollution")
parser.add_argument("--k1", type=int,default=1,help = "Number of desired types(choice [1,2,3,5])")
parser.add_argument("--k2", type=int,default=5,help = "Number of pollution types(choice [1,2,3,5])")
parser.add_argument("--lambda1", type=int,default=1,help="lambda")
parser.add_argument("--c",type=float,default=0.5,help="c value")
parser.add_argument("--theorem",type=int,default=1,help="theorem 1 or theorem 2")
opt = parser.parse_args()
print(opt)

np.random.seed(opt.seed)

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu



if(opt.theorem==1):
    bn = 0
else:
    pi = (1-opt.gamma_p)
    bn = (2*pi-1)/(1+pi)
c = opt.c

from inc3 import InceptionV3
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model_inv = InceptionV3([block_idx])


from testing import calculate_activation_statistics





class Generator(nn.Cell):
    def __init__(self,  nc=1, nz=100, ngf=64):
        super(Generator, self).__init__()
        
        self.main = nn.SequentialCell(
            
            nn.Conv2dTranspose(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.Conv2dTranspose2(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.Conv2dTranspose(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
        
            nn.Conv2dTranspose(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2dTranspose(    ngf,      nc, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def construct(self, input):
        output = self.main(input)
        return output


class Discriminator(nn.Cell):
    def __init__(self, nc=1, ngf=64):
        super().__init__()

  
        self.conv1 = nn.Conv2d(nc, ngf,
            4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ngf,ngf*2,
            4, 2, 1, bias=False)


        self.conv3 = nn.Conv2d(ngf*2, ngf*4,
            4, 2, 1, bias=False)
        self.conv4 = nn.Conv2d(ngf*4, 1, 4, 1, 0, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)
    def construct(self, x):
        x = self.leaky_relu(self.conv1(x), 0.2, True)
        x = self.leaky_relu((self.conv2(x)), 0.2, True)
        x = self.leaky_relu((self.conv3(x)), 0.2, True)
       
        x = self.conv4(x)

        

        return x

PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))



adversarial_loss = nn.MSELoss()

generator = Generator()
discriminator = Discriminator()


from dataset.mnistset import create_loader
train_pos, train_neg, test_eval = create_loader(opt)

optimizer_G = nn.Adam(generator.trainable_params(), learning_rate=0.0001, beta1=0.5, beta2=0.9)
optimizer_D = nn.Adam(discriminator.trainable_params(), learning_rate=0.0001, beta1=0.5, beta2=0.9)
optimizer_G.update_parameters_name('optim_g')
optimizer_D.update_parameters_name('optim_d')

   





real_images_all = []
with torch.no_grad():
    start = time.time()
    for idx, (image, target) in enumerate(test_eval):
        image = image.to(device)
        
        image = image.expand(-1,3,-1,-1)
        target = target.to(device)
        
        
       
        real_images_all.append(image)
        
    real_images_all = torch.cat(real_images_all)
   
  
    real_images_all = real_images_all.to(device)
    mu1, sigma1 = calculate_activation_statistics(real_images_all,model_inv)
    end = time.time()

auc_re = pd.DataFrame()


auc_re = pd.DataFrame()
from testing import testing 
batches_done=0
best_fid=100000
for epoch in range(opt.n_epochs):
    discriminator.train()
    generator.train()
    i=0
    
    for (batch_pos,batch_neg) in zip(train_pos,cycle(train_neg)):
        discriminator.train()
        generator.train()
        i+=1
        batches_done+=1
    
        
        img_pos = batch_pos[0]
        img_neg = batch_neg[0]
        img_pos = img_pos.to(device)
        img_neg = img_neg.to(device)
        target_pos = batch_pos[1]
        target_neg = batch_neg[1]
        
        optimizer_D.zero_grad()
        

        valid = Variable(Tensor(img_pos.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(img_pos.shape[0], 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(img_pos.type(Tensor))
        z = Variable(Tensor(np.random.normal(0, 1, (img_pos.shape[0], opt.latent_dim,1,1))))

        gen_imgs = generator(z)
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(img_pos).squeeze(), a*valid.squeeze())
        neg_loss = adversarial_loss(discriminator(img_neg).squeeze(),bn*(torch.ones([img_neg.size(0), 1])).to(device).squeeze())
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()).squeeze(), b*valid.squeeze())
        
           
        d_loss = real_loss + opt.lambda1* neg_loss + fake_loss 
        
        d_loss.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()

        
        
        g_loss = adversarial_loss(discriminator(img_pos), c*valid) +adversarial_loss(discriminator(gen_imgs), c*valid)+adversarial_loss(discriminator(img_neg),c*(torch.ones([img_neg.size(0),1])).to(device))
        
        g_loss.backward()
        optimizer_G.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_pos), d_loss.item(), g_loss.item())
        )
        
        
        
        
        
        
        
        
        
        if(batches_done%500==0):
            df_dict = testing(model_inv,mu1,sigma1,generator,batches_done,opt,device)
            if(df_dict['fid']<best_fid):
                best_fid=df_dict['fid']
            print("[Best_fid: %f] "% (best_fid))
            auc_re = auc_re.append(df_dict,ignore_index=True)
if not os.path.exists(PACK_PATH +opt.dir):
    os.makedirs(PACK_PATH+opt.dir)
auc_re.to_csv(PACK_PATH+opt.dir+opt.name+"seed_"+str(opt.seed)+".csv")
