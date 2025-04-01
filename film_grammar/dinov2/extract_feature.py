# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import ipdb

import torch
from torch.nn.functional import one_hot, softmax

import dinov2.distributed as distributed
from dinov2.data import SamplerType, make_data_loader, make_dataset
from dinov2.data.transforms import make_classification_eval_transform
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers

from PIL import Image
from torchvision import transforms


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = [],
    add_help: bool = True,
):
    setup_args_parser = get_setup_args_parser(parents=parents, add_help=False)
    parents = [setup_args_parser]
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--train-dataset",
        dest="train_dataset_str",
        type=str,
        help="Training dataset",
    )
    parser.add_argument(
        "--val-dataset",
        dest="val_dataset_str",
        type=str,
        help="Validation dataset",
    )
    parser.add_argument(
        "--nb_knn",
        nargs="+",
        type=int,
        help="Number of NN to use. 20 is usually working the best.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature used in the voting coefficient",
    )
    parser.add_argument(
        "--gather-on-cpu",
        action="store_true",
        help="Whether to gather the train features on cpu, slower"
        "but useful to avoid OOM for large datasets (e.g. ImageNet22k).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size.",
    )
    parser.add_argument(
        "--n-per-class-list",
        nargs="+",
        type=int,
        help="Number to take per class",
    )
    parser.add_argument(
        "--n-tries",
        type=int,
        help="Number of tries",
    )
    parser.set_defaults(
        train_dataset_str="ImageNet:split=TRAIN",
        val_dataset_str="ImageNet:split=VAL",
        nb_knn=[10, 20, 100, 200],
        temperature=0.07,
        batch_size=256,
        n_per_class_list=[-1],
        n_tries=1,
    )
    return parser


def dynamic_aspect_ratio_crop(img):
    # Get the original dimensions of the image
    width, height = img.size
    
    # Set aspect ratio based on image orientation
    if width > height:
        aspect_ratio = 4 / 3  # For landscape images
    else:
        aspect_ratio = 3 / 4  # For portrait images
    
    # Calculate the new crop dimensions
    if width / height > aspect_ratio:
        # Crop width to match the desired aspect ratio
        new_width = int(height * aspect_ratio)
        new_height = height
    else:
        # Crop height to match the desired aspect ratio
        new_width = width
        new_height = int(width / aspect_ratio)
    
    # Center crop the image
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    # Perform the crop
    return img.crop((left, top, right, bottom))

class DINOv2_feature_extractor():
    def __init__(self, model_type, aspect_ratio):
        if model_type == "vitb14":
            pretrained_weights = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
            config_file = "dinov2/dinov2/configs/eval/vitb14_pretrain.yaml"
        elif model_type == "vitl14_reg4":
            pretrained_weights = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth"
            config_file = "dinov2/dinov2/configs/eval/vitl14_reg4_pretrain.yaml"
        elif model_type == "vitg14_reg4":
            pretrained_weights = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth"
            config_file = "dinov2/dinov2/configs/eval/vitg14_reg4_pretrain.yaml"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.building_dino_v2(pretrained_weights, config_file).to(self.device)
        self.model.eval()
        

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: dynamic_aspect_ratio_crop(img)),  # Central crop based on dynamic aspect ratio
            transforms.Lambda(lambda img: img.resize(
                (int(224 * aspect_ratio), 224) if img.width < img.height else (224, int(224 * aspect_ratio)), 
                Image.BICUBIC)
            ), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        
    def building_dino_v2(self, pretrained_weights, config_file):
        args_parser_new = get_args_parser()
        args_new = args_parser_new.parse_args()
        args_new.pretrained_weights = pretrained_weights
        args_new.config_file = config_file
        model, autocast_dtype = setup_and_build_model(args_new)
        n_last_blocks_list = [1] #[1, 4]
        n_last_blocks = max(n_last_blocks_list)
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)
        dino_v2_feat_model = ModelWithIntermediateLayers(model, n_last_blocks, autocast_ctx)
        return dino_v2_feat_model

    def extract_dino_v2_feat_from_tensor(self, transformed_imgs):
        # Input: a tensor starting with batch dimension
        sample_output = self.model(transformed_imgs)
        spatial_feats, cls_feats = sample_output[0]
        concat_feats = torch.cat([cls_feats.unsqueeze(1), spatial_feats], 1)
        return concat_feats

    def extract_dino_v2_feat_from_pil(self, imgs):
        # Input: a list of PIL images
        transformed_imgs = []
        for img in imgs:
            transformed_img = self.transform(img).to(self.device)
            transformed_imgs.append(transformed_img)
        transformed_imgs = torch.stack(transformed_imgs, 0)  # B C H W
        sample_output = self.model(transformed_imgs)
        spatial_feats, cls_feats = sample_output[0]
        concat_feats = torch.cat([cls_feats.unsqueeze(1), spatial_feats], 1)
        return concat_feats


    


if __name__ == "__main__":
    description = "DINOv2 feature extractor"
    args_parser = get_args_parser(description=description)
    args = args_parser.parse_args()

