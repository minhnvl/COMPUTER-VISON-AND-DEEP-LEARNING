#!/usr/bin/env python
# -*- coding: utf-8 -*-
from preprocessing import full_prep
from config_submit import config as config_submit

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable

from layers import acc
from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier

from utils import *
from split_combine import SplitComb
from test_detect import test_detect
from importlib import import_module
import pandas
import numpy as np


def main_init():
    datapath = config_submit['datapath']
    prep_result_path = config_submit['preprocess_result_path']
    skip_prep = config_submit['skip_preprocessing']
    skip_detect = config_submit['skip_detect']
    if not skip_prep:
        testsplit = full_prep(datapath,prep_result_path,
                            n_worker = config_submit['n_worker_preprocessing'],
                            use_existing=config_submit['use_exsiting_preprocessing'])
    else:
        print(datapath)
        testsplit = os.listdir(datapath)
    print("------Get Model---------")
    nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
    config1, nod_net, loss, get_pbb = nodmodel.get_model()
    checkpoint = torch.load(config_submit['detector_param'])
    nod_net.load_state_dict(checkpoint['state_dict'])
    torch.cuda.set_device(1)
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(1))

    device = torch.device("cuda:0")

    nod_net = nod_net.cuda(device)
    cudnn.benchmark = True
    nod_net = DataParallel(nod_net)
    # print(nod_net)
    bbox_result_path = config_submit['bbox_result']
    if not os.path.exists(bbox_result_path):
        os.mkdir(bbox_result_path)
    #testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]
    print(config1)
    if not skip_detect:
        margin = 32
        sidelen = 144
    #    config1['datadir'] = prep_result_path
        split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value= config1['pad_value'])

        dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
        test_loader = DataLoader(dataset,batch_size = 32,
            shuffle = False,num_workers = 32,pin_memory=False,collate_fn =collate)
        print(get_pbb)
        test_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])
    print("-----Finish--------")






# print("-----Get Classification--------")
# casemodel = import_module(config_submit['classifier_model'].split('.py')[0])
# casenet = casemodel.CaseNet(topk=5)
# config2 = casemodel.config
# print(config2)
# print(config_submit['classifier_param'])
# f = open(config_submit['classifier_param'],'rb')
# checkpoint = torch.load(f,encoding='latin')
# casenet.load_state_dict(checkpoint['state_dict'])
# torch.cuda.set_device(1)
# device = torch.device("cuda:0")
# casenet = casenet.cuda(device)
# cudnn.benchmark = True
# casenet = DataParallel(casenet)




# checkpoint = torch.load(config_submit['detector_param'])
# nod_net.load_state_dict(checkpoint['state_dict'])
# torch.cuda.set_device(1)
# print(torch.cuda.device_count())
# print(torch.cuda.get_device_name(1))
# device = torch.device("cuda:0")
# config2['bboxpath'] = bbox_result_path
# config2['datadir'] = prep_result_path


# print("---Test Data---------")
# dataset = DataBowl3Classifier(testsplit, config1, phase = 'test')
# predlist = test_casenet(nod_net,dataset).T
# filename = config_submit['outputfile']
# df = pandas.DataFrame({'id':testsplit, 'cancer':predlist})
# df.to_csv(filename,index=False)


# def test_casenet(model,testset):
#     data_loader = DataLoader(
#         testset,
#         batch_size = 1,
#         shuffle = False,
#         num_workers = 32,
#         pin_memory=True)
#     #model = model.cuda()
#     model.eval()
#     predlist = []
    
#     #     weight = torch.from_numpy(np.ones_like(y).float().cuda()
#     for i,(x,coord) in enumerate(data_loader):

#         coord = Variable(coord).cuda()
#         x = Variable(x).cuda()
#         nodulePred,casePred,_ = model(x,coord)
#         predlist.append(casePred.data.cpu().numpy())
#         #print([i,data_loader.dataset.split[i,1],casePred.data.cpu().numpy()])
#     predlist = np.concatenate(predlist)
#     return predlist    
