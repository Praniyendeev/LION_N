# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

""" copied and modified from https://github.com/stevenygd/PointFlow/blob/master/datasets.py """
import os
import open3d as o3d
import time
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from torch.utils import data
import random
import tqdm
from datasets.data_path import get_path
from PIL import Image
import plotly
import navis
import pickle

OVERFIT = 0

# taken from https://github.com/optas/latent_3d_points/blob/
# 8e8f29f8124ed5fc59439e8551ba7ef7567c9a37/src/in_out.py

class NeuronPointClouds(Dataset):
    def __init__(self,
                 tr_sample_size=10000,
                 te_sample_size=10000,
                 split='train',
                 splits={"train":0.8,"val":0.2},
                 scale=1.,
                 standardize=True,
                 normalize_global=True,
                 recenter_per_shape=False,
                 all_points_mean=None,
                 all_points_std=None,
                 input_dim=3, 
                 ):
        print("##############################################################################")
        print("##############################################################################")
        print("##############################################################################")
        print("##############################################################################")
       
        
        
        #attributes
        self.tr_sample_size =tr_sample_size
        self.normalize_global = normalize_global
        self.standardize = standardize
        self.display_axis_order = [0, 1, 2]
        self.all_points_mean = 0
        self.all_points_std = 1

        #path init
        swc_path = "./hemibrain/raw_swc"
        swc_path = "/mnt/nvme/node03/pranav/CellType/data/hemibrain/raw_swc"
        if os.path.exists(swc_path+f"/{split}_split.pkl"):
            with open(swc_path+f"/{split}_split.pkl",'rb') as f:
                self.swc_list = pickle.load(f)
        else:
            
            self.swc_list =[sp for sp in os.listdir(swc_path) if os.path.isfile(swc_path+"/"+sp)]
            offset = 0
            for split_name, split_val in splits.items():
                split_len = int( len(self.swc_list)*split_val)
                swc_split = self.swc_list[offset:offset+split_len]
                with open(swc_path+f"/{split_name}_split.pkl",'wb') as f:
                    pickle.dump(swc_split,f)
                    offset += split_len

        self.swc_path=swc_path

        # process_batch_size = 100
        # for i in range(0,len(self.neuron_list),process_batch_size):
        #     neuron_batch = self.neuron_list[i:i+process_batch_size]
        #     swc_neuron=navis.read_swc(neuron_batch)
        #     point_neuron=navis.make_dotprops(swc_neuron)


    def __len__(self):
        return len(self.swc_list)


    def __getitem__(self, index) -> dict:
        swc_neuron=navis.read_swc(self.swc_path+"/"+self.swc_list[index])
        point_neuron=navis.make_dotprops(swc_neuron)
        idx=np.random.choice(len(point_neuron.points),self.tr_sample_size)
        if len(point_neuron.points) > self.tr_sample_size :
            point_cloud_neuron =o3d.geometry.PointCloud(  o3d.utility.Vector3dVector(point_neuron.points))
            point_cloud_neuron=point_cloud_neuron.farthest_point_down_sample(self.tr_sample_size)
            point_tensor =torch.Tensor(point_cloud_neuron.points) #[None,:,:]
        else:
            # idx=np.random.choice(len(point_neuron.points),self.tr_sample_size)
            point_tensor =torch.from_numpy(point_neuron.points[idx])# [None,:,:]

        if self.standardize:
            point_tensor=(point_tensor-point_tensor.min())/(point_tensor.max()-point_tensor.min())
            point_tensor = 2 * point_tensor -1
        
        if self.normalize_global:
            point_tensor=(point_tensor - self.all_points_mean) / \
            self.all_points_std
        
        # tr_idxs = np.arange(self.tr_sample_size)
        # tr_out = torch.from_numpy(tr_out[tr_idxs, :]).float()
        m, s = self.all_points_mean, self.all_points_std

        # cate_idx = self.cate_idx_lst[idx]
        # sid, mid = self.all_cate_mids[idx]
    
        output={
                'idx': index,
                'select_idx': idx,
                'tr_points': point_tensor,
                'input_pts': point_tensor,
                'mean': m,
                'std': s,
                'cate_idx': torch.zeros(1),
                'sid': self.swc_list[index].split(".")[0],
                'mid': "NONE",
                'display_axis_order': self.display_axis_order
            }


        return output




def init_np_seed(worker_id):
    seed = torch.initial_seed()
    np.random.seed(seed % 4294967296)


def get_datasets(cfg, args):
    """
        cfg: config.data sub part 
    """
    if OVERFIT:
        random_subsample = 0
    else:
        random_subsample = cfg.random_subsample
    logger.info(f'get_datasets: tr_sample_size={cfg.tr_max_sample_points}, '
                f' te_sample_size={cfg.te_max_sample_points}; '
                f' random_subsample={random_subsample}'
                f' normalize_global={cfg.normalize_global}'
                f' normalize_std_per_axix={cfg.normalize_std_per_axis}'
                f' normalize_per_shape={cfg.normalize_per_shape}'
                f' recenter_per_shape={cfg.recenter_per_shape}'
                )
    kwargs = {}
    tr_dataset = NeuronPointClouds(
        split='train',
        tr_sample_size=cfg.tr_max_sample_points,
        scale=cfg.dataset_scale,
        normalize_global=cfg.normalize_global,
        **kwargs

    )
    te_dataset = NeuronPointClouds(
        split='val',
        tr_sample_size=cfg.tr_max_sample_points,
        scale=cfg.dataset_scale,
        normalize_global=cfg.normalize_global,
        **kwargs

    )

    return tr_dataset, te_dataset


def get_data_loaders(cfg, args):
    tr_dataset, te_dataset = get_datasets(cfg, args)
    kwargs = {}
    if args.distributed:
        kwargs['sampler'] = data.distributed.DistributedSampler(
            tr_dataset, shuffle=True)
    else:
        kwargs['shuffle'] = True
    if args.eval_trainnll:
        kwargs['shuffle'] = False
    train_loader = data.DataLoader(dataset=tr_dataset,
                                   batch_size=cfg.batch_size,
                                   num_workers=cfg.num_workers,
                                   drop_last=cfg.train_drop_last == 1,
                                   pin_memory=False, **kwargs)
    test_loader = data.DataLoader(dataset=te_dataset,
                                  batch_size=cfg.batch_size_test,
                                  shuffle=False,
                                  num_workers=cfg.num_workers,
                                  pin_memory=False,
                                  drop_last=False,
                                  )
    logger.info(
        f'[Batch Size] train={cfg.batch_size}, test={cfg.batch_size_test}; drop-last={cfg.train_drop_last}')
    loaders = {
        "test_loader": test_loader,
        'train_loader': train_loader,
    }
    return loaders
