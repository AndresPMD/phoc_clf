# -*- coding: utf-8 -*-

"""
    Fine-grained Classification based on textual cues
"""

# Python modules
import torch

import torch.nn as nn
import time
import torch
import numpy as np
import glob
import os
import json
from PIL import Image, ImageDraw, ImageFile

import torchvision
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm

import pdb
import sys

# Own modules
from logger import LogMetric
from utils import *
from options import *
from data.data_generator import *
from models.models import load_model
from custom_optim import *
__author__ = "Andres Mafla Delgado; Sounak Dey"
__email__ = "amafla@cvc.uab.cat; sdey@cvc.uab.cat"


def test(args, net, cuda):

    processed_imgs = 0
    # Switch to evaluation mode
    net.eval()

    if not os.path.exists(args.base_dir+'proxy_features/'):
        os.mkdir(args.base_dir+'proxy_features/')

    with torch.no_grad():
        image_list = []
        with open ('/tmp-network/user/amafla/data/cocotext_with_captions/image_list.txt', 'r') as fp:
            image_list = fp.readlines()
        fusioned_feats = np.zeros([len(image_list), 512])
        for idx, image in tqdm(enumerate(image_list)):
            # Load Image features
            img_feats_path = args.base_dir + 'embeddings/' + image.split('.')[0] + '.json'
            with open (img_feats_path, 'r') as fp:
                img_feats = json.load(fp)
            img_feats = np.asarray(img_feats)
            # Load PHOC_FV features
            phoc_FV_path = args.base_dir + 'phoc_FV_features/' + image.split('.')[0] + '.json'
            with open(phoc_FV_path, 'r') as fp:
                textual_feats = json.load(fp)
            textual_feats = np.asarray(textual_feats)
            img_feats = torch.from_numpy(img_feats)
            img_feats = img_feats.type(torch.FloatTensor)
            textual_feats = torch.from_numpy(textual_feats)
            textual_feats = textual_feats.type(torch.FloatTensor)

            if cuda:
                img_feats, textual_feats = img_feats.cuda(), textual_feats.cuda()
            img_feats = Variable(img_feats)
            textual_feats = Variable(textual_feats)

            img_feats = img_feats.view(-1, 512)
            textual_feats = textual_feats.view(-1, 38400)
            output, fusion_vec_norm = net(img_feats, textual_feats, sample_size=1)

            features = fusion_vec_norm.cpu().numpy()
            fusioned_feats[idx] = features
            features = features.tolist()
            with open (args.base_dir + 'proxy_features/' + image.split('.')[0] + '.json', 'w') as fp:
                json.dump(features, fp)
    # Save as numpy features
    np.save('/tmp-network/user/amafla/data/cocotext_with_captions/proxy_features.npy', fusioned_feats)


    print ('Complete!')
    return

def main():
    print('Preparing data')

    if args.dataset =='cocotext':
        # Similar Number of classes as Con-Text (Proxy Task)
        num_classes = 28
        weight_file = '/tmp-network/user/rsampaio/models/finegrained-classif/context_RMAC_Full_fisher_yolo_phoc_mlb_ep3.0/checkpoint_mlb_3.0_L1_0.7800202876657005.weights'
    else:
        print('Dataset error')

    embedding_size = get_embedding_size(args.embedding)
    print('Loading Model')
    net = load_model(args, num_classes, embedding_size)
    checkpoint = load_checkpoint(weight_file)
    net.load_state_dict(checkpoint)

    print('Checking CUDA')
    if args.cuda and args.ngpu > 1:
        print('\t* Data Parallel **NOT TESTED**')
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.cuda:
        print('\t* CUDA ENABLED!')
        net = net.cuda()

    print('\n*** TEST ***\n')
    test(args, net, args.cuda)
    print('*** Feature Extraction Completed ***')
    sys.exit()

if __name__ == '__main__':
    # Parse options
    args = Options_Test().parse()
    print('Parameters:\t' + str(args))

    # Check cuda & Set random seed
    args.cuda = args.ngpu > 0 and torch.cuda.is_available()
    main()