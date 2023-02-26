# -*- coding: utf-8 -*-


import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms
import numpy as np
import mindspore as ms
import mindspore.dataset.vision as vision

def create_loader(opt,kwargs):
    print('--load mnist dataset--')    
    
    anomaly_list = list(np.arange(0,10))
    random_list = np.arange(0,10)
    np.random.shuffle(random_list)
    
    data1 = np.load('train_data_mnist.npy')
    target1 = np.load('train_label_mnist.npy')
    
    
    
    if(opt.k1==1):
        data1_p = data1[target1==anomaly_list[random_list[0]]]
        target1_p = target1[target1==anomaly_list[random_list[0]]]
    elif(opt.k1==2):
        data1_p = data1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])]
        target1_p = target1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])]
    elif(opt.k1 == 3):
        data1_p = data1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])]
        target1_p = target1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])]
    else:
        data1_p = data1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])|(target1==anomaly_list[random_list[3]])|(target1==anomaly_list[random_list[4]])]
        target1_p = target1[(target1==anomaly_list[random_list[0]])|(target1==anomaly_list[random_list[1]])|(target1==anomaly_list[random_list[2]])|(target1==anomaly_list[random_list[3]])|(target1==anomaly_list[random_list[4]])]
    randIdx_normal = np.arange(data1_p.shape[0])
    np.random.shuffle(randIdx_normal)
    
    if(opt.k2==1):
        data1_n = data1[target1==anomaly_list[random_list[5]]]
        target1_n = target1[target1==anomaly_list[random_list[5]]]
    elif(opt.k2==2):
        data1_n = data1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])]
        target1_n = target1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])]
    elif(opt.k2 == 3):
        data1_n = data1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])]
        target1_n = target1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])]
    else:
        data1_n = data1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])|(target1==anomaly_list[random_list[8]])|(target1==anomaly_list[random_list[9]])]
        target1_n = target1[(target1==anomaly_list[random_list[5]])|(target1==anomaly_list[random_list[6]])|(target1==anomaly_list[random_list[7]])|(target1==anomaly_list[random_list[8]])|(target1==anomaly_list[random_list[9]])]
    
    
    randIdx = np.arange(data1_n.shape[0])
    np.random.shuffle(randIdx)
    num1 = 6000
    
    data1 = np.concatenate((data1_p[randIdx_normal[:num1]],data1_n[randIdx[:int(opt.gamma_p*num1//(1-opt.gamma_p))]]),axis=0)
    target1 = np.concatenate((target1_p[randIdx_normal[:num1]],target1_n[randIdx[:int(opt.gamma_p*num1//(1-opt.gamma_p))]]),axis=0)
    train_pos = ds.NumpySlicesDataset((data1, target1), ["data", "label"],shuffle=True)
    #train_pos = train_pos.map(operations=vision.Resize(size=(32, 32)), input_columns="data")
    train_pos = train_pos.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    train_pos = train_pos.batch(batch_size=opt.batch_size,drop_remainder=False)
    
    num2 = data1_n.shape[0]
    
    data2 = data1_n[randIdx[num2-int(num1//(1/opt.gamma_c)):num2]]
    target2 = target1_n[randIdx[num2-int(num1//(1/opt.gamma_c)):num2]]
    
    
    randIdx = np.arange(data2.shape[0])
    np.random.shuffle(randIdx)
    
    train_neg = ds.NumpySlicesDataset((data2, target2), ["data", "label"],shuffle=True) 
    train_neg = train_neg.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    train_neg = train_neg.batch(batch_size=opt.batch_size//9,drop_remainder=False)
   
    data4 = np.load('test_data_mnist.npy')
    target4 = np.load('test_label_mnist.npy')
    
    
    if(opt.k1==1):
        data4_p = data4[target4==anomaly_list[random_list[0]]]
        target4_p = target4[target4==anomaly_list[random_list[0]]]
    elif(opt.k1==2):
        data4_p = data4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])]
        target4_p = target4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])]
    elif(opt.k1 == 3):
        data4_p = data4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])]
        target4_p = target4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])]
    else:
        data4_p = data4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])|(target4==anomaly_list[random_list[3]])|(target4==anomaly_list[random_list[4]])]
        target4_p = target4[(target4==anomaly_list[random_list[0]])|(target4==anomaly_list[random_list[1]])|(target4==anomaly_list[random_list[2]])|(target4==anomaly_list[random_list[3]])|(target4==anomaly_list[random_list[4]])]
    randIdx_test = np.arange(data4_p.shape[0])
    data4 = data4_p[randIdx_test[:5000]]
    target4 = target4_p[randIdx_test[:5000]]
    test_loader = ds.NumpySlicesDataset((data4, target4), ["data", "label"],shuffle=True) 
    test_loader = test_loader.map(operations=transforms.TypeCast(ms.int32), input_columns="label")
    test_loader = test_loader.batch(batch_size=opt.batch_size//9,drop_remainder=False)
    return train_pos,train_neg,test_loader
