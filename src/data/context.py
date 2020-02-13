#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data
from torchvision import transforms
import os
import pickle
import random
import sys
import json
from PIL import Image

sys.path.insert(0, '.')

import numpy as np
from skimage import io
import pdb


def Context_dataset(args, embedding_size):
    # Random seed
    np.random.seed(args.seed)

    # Getting the classes and annotations
    # ******
    data_path = args.data_path
    with open(data_path+'/Context/data/split_'+ str(args.split) +'.json','r') as fp:
        gt_annotations = json.load(fp)

    # Load Embedding according to OCR
    if args.embedding == 'w2vec' or args.embedding == 'fasttext' or args.embedding == 'glove' or args.embedding =='bert':
        with open(data_path + '/Context/' + args.ocr + '/text_embeddings/Context_' + args.embedding + '.pickle','rb') as fp:
            text_embedding = pickle.load(fp)
    elif args.embedding =='phoc':
        text_embedding = {'embedding':'phoc'}
    elif args.embedding == 'fisher':
        text_embedding = {'embedding':'fisher'}
    else:
        print('OCR SELECTED NOT IMPLEMENTED')
    # Data Loaders
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = Context_Train(args, gt_annotations, text_embedding, embedding_size, train_transform)
    test_loader = Context_Test(args, gt_annotations, text_embedding, embedding_size, test_transform)

    return train_loader, test_loader, gt_annotations, text_embedding


class Context_Train(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, transform=None):

        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['train'].keys())
        #Random.shuffle(self.image_list)

    def __len__(self):
        return len(self.gt_annotations['train'])

    def __getitem__(self, index):
        data_path = self.args.data_path
        assert index <= len(self), 'index range error'
        image_name = self.image_list[index].rstrip()
        image_path = data_path+'/Context/data/JPEGImages/' + image_name
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_class = self.gt_annotations['train'][image_name]
        label = np.zeros(28)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        # LOAD RMAC FEATURES
        if self.args.img_embs == 'rmac':
            img_embs_path = os.path.join(data_path, 'embeddings')
        elif self.args.img_embs == 'vse':
            img_embs_path = os.path.join(data_path, 'vsepp')
        with open (os.path.join(img_embs_path, image_name[:-3]+'json'),'r') as fp:
            rmac_feats = json.load(fp)
        rmac_feats = np.asarray(rmac_feats)
        rmac_feats = torch.from_numpy(rmac_feats)
        rmac_feats = rmac_feats.type(torch.FloatTensor)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding == 'glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name])
        elif self.args.embedding == 'phoc':
            with open (data_path + '/Context/yolo_phoc/'+image_name[:-3]+'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))
        elif self.args.embedding == 'fisher':
            if self.args.ocr == 'yolo_phoc':
                relative_path = '/Context/old_fisher_vectors/'
            elif self.args.ocr == 'e2e_mlt':
                relative_path = '/Context/fasttext_fisher/'
            else: print('Not Implemented')
            with open (data_path + relative_path +image_name[:-3]+'json')as fp:
                fisher_vector = json.load(fp)
                text_embedding = np.resize(fisher_vector, (1, 38400))
        # FISHER VECTORS DO NOT NEED MAX TEXTUAL
        if self.args.embedding != 'fisher':
            text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(text_embedding)[0] == 0:
                text_embedding = np.zeros((1,self.embedding_size))
            elif np.shape(text_embedding)[0] > self.args.max_textual:
                text_embedding = text_embedding[0:self.args.max_textual]
            text_features[:len(text_embedding)] = text_embedding
        else:
            text_features = text_embedding

        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)

        return img, label, text_features, rmac_feats

class Context_Test(data.Dataset):
    def __init__(self, args, gt_annotations, text_embedding, embedding_size, transform=None):
        self.args = args
        self.gt_annotations = gt_annotations
        self.text_embedding = text_embedding
        self.embedding_size = embedding_size
        self.transform = transform
        self.image_list = list(gt_annotations['test'].keys())

    def __len__(self):
        return len(self.gt_annotations['test'])

    def __getitem__(self, index):
        data_path = self.args.data_path
        assert index <= len(self), 'index range error'
        image_name = self.image_list[index].rstrip()
        image_path = data_path+ '/Context/data/JPEGImages/' + image_name
        img = Image.open(image_path).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_class = self.gt_annotations['test'][image_name]
        label = np.zeros(28)
        label[int(img_class) - 1] = 1
        label = torch.from_numpy(label)
        label = label.type(torch.FloatTensor)

        with open('/tmp-network/user/amafla/data/Context/data/embeddings/' + image_name[:-3] + 'json', 'r') as fp:
            rmac_feats = json.load(fp)
        rmac_feats = np.asarray(rmac_feats)
        rmac_feats = torch.from_numpy(rmac_feats)
        rmac_feats = rmac_feats.type(torch.FloatTensor)

        if self.args.embedding == 'w2vec' or self.args.embedding == 'fasttext' or self.args.embedding == 'glove' or self.args.embedding == 'bert':
            text_embedding = np.asarray(self.text_embedding[image_name])
        elif self.args.embedding == 'phoc':
            with open (data_path + '/Context/yolo_phoc/'+image_name[:-3]+'json') as fp:
                phocs = json.load(fp)
                text_embedding = np.resize(phocs, (np.shape(phocs)[0], 604))

        elif self.args.embedding == 'fisher':
            if self.args.ocr == 'yolo_phoc':
                relative_path = '/Context/old_fisher_vectors/'
            elif self.args.ocr == 'e2e_mlt':
                relative_path = '/Context/fasttext_fisher/'
            else: print('Not Implemented')
            with open (data_path + relative_path +image_name[:-3]+'json')as fp:
                fisher_vector = json.load(fp)
                text_embedding = np.resize(fisher_vector, (1, 38400))
        # FISHER VECTORS DO NOT NEED MAX TEXTUAL
        if self.args.embedding != 'fisher':
            text_features = np.zeros((self.args.max_textual, self.embedding_size))
            if np.shape(text_embedding)[0] == 0:
                text_embedding = np.zeros((1,self.embedding_size))
            elif np.shape(text_embedding)[0] > self.args.max_textual:
                text_embedding = text_embedding[0:self.args.max_textual]
            text_features[:len(text_embedding)] = text_embedding
        else:
            text_features = text_embedding

        text_features = torch.from_numpy(text_features)
        text_features = text_features.type(torch.FloatTensor)

        return img, label, text_features, rmac_feats

