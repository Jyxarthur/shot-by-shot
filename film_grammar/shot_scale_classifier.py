import os
import ast
import sys
import copy
import torch
import tarfile
import argparse
import torch.nn as nn
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import torch.cuda.amp as amp 
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from functools import partial
from typing import List, Optional
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader

sys.path.append("dinov2")
from dinov2.eval.setup import get_args_parser as get_setup_args_parser
from dinov2.eval.setup import setup_and_build_model
from dinov2.eval.utils import ModelWithIntermediateLayers

from io import BytesIO
from decord import VideoReader, cpu    



# This is the default args parser in DINOv2, unchanged
def get_args_parser_dinov2(
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


class ScaleClassifier(nn.Module):
    def __init__(self, feature_model, autocast_dtype, dim=768, label=5):
        super().__init__()
        self.feature_model = feature_model
        self.linear = nn.Linear(dim, label)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=autocast_dtype)

    def forward(self, images, inference=False):
        n_last_blocks = 1
        with torch.inference_mode():
            with self.autocast_ctx():
                features = self.feature_model.get_intermediate_layers(
                    images, n_last_blocks, return_class_token=True
                )
        output = self.linear(features[0][1])
        return output


class ScaleDataset():
    def __init__(self,
                dataset, 
                anno_df,
                video_root,
                **kwargs):
        self.dataset = dataset
        self.video_root = video_root

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        df = pd.read_csv(anno_df)
        self.samples = []
        for row_idx, row in tqdm(df.iterrows(), total=len(df)):
            if self.dataset == "cmdad":
                cmd_filename = row['cmd_filename']
                video_path = os.path.join(self.video_root, cmd_filename + '.mkv')
                start = row['scaled_start']  
                end = row['scaled_end'] 
            elif self.dataset == "madeval":
                imdbid = row["imdbid"]
                video_path = os.path.join(self.video_root, imdbid + '.mkv')
                start = row['scaled_start']  
                end = row['scaled_end'] 
            elif self.dataset == "tvad":
                seg_name = row["tvad_name"]
                if "friends" in seg_name:
                    video_path = os.path.join(self.video_root, "friends_frames", seg_name + ".tar")
                else:
                    video_path = os.path.join(self.video_root, "bbt_frames", seg_name + ".tar")
                start = row['scaled_index']  
                end = None
            self.samples.append((row_idx, row['anno_idx'], video_path, start, end))

        if self.dataset == "madeval": # to save time for madeval load
            self.current_decord = None
            self.current_video_path = None

    def __getitem__(self, index):
        row_idx, anno_idx, video_path, start, end = self.samples[index]

        # Extract the middle frame (out of 32 frames) of each shot; "clip" here means "video clip"
        if self.dataset == "tvad":
            scaled_indices = ast.literal_eval(start)
            clip_image = extract_midframe_from_tarfile(video_path, int(scaled_indices[len(scaled_indices) // 2]))
        elif self.dataset == "madeval":
            if self.current_video_path is None or self.current_video_path != video_path:
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0))
                self.current_decord = decord_vr
                self.current_video_path = video_path
            else:
                decord_vr = self.current_decord
            clip_image = extract_midframe_from_video(decord_vr, start = start, end = end)[0]  
        elif self.dataset == "cmdad":
            decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
            clip_image = extract_midframe_from_video(decord_vr, start = start, end = end)[0]  
        
        # Crop the center 4(width):3(height) region of the image
        w_, h_ = clip_image.size
        if w_ / h_ > 4 / 3:
            new_h_ = 336
            new_w_ = int(new_h_ / h_ * w_)
            left = (new_w_ - 448) // 2
            crop_box = (left, 0, left + 448, 336)            
        else:
            new_w_ = 448
            new_h_ = int(new_w_ / w_ * h_)
            top = (new_h_ - 336) // 2
            crop_box = (0, top, 448, top + 336)
        clip_image = clip_image.resize((new_w_, new_h_))
        clip_image = clip_image.crop(crop_box)
        clip_image_tensor = self.transform(clip_image)

        return_dict = {"row_idx": row_idx, \
                    "anno_idx": anno_idx, \
                    "image_tensor": clip_image_tensor
        }
        return return_dict
        
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['row_idx'] = [sample['row_idx'] for sample in batch]
        out_batch['anno_idx'] = [sample['anno_idx'] for sample in batch]
        out_batch['image_tensor'] = torch.stack([sample['image_tensor'] for sample in batch], 0)
        return out_batch

    def __len__(self):
        return len(self.samples)

       

def extract_midframe_from_tarfile(tar_path, img_idx):
    dirname = os.path.basename(tar_path.replace(".tar", ""))
    imagename = os.path.join(dirname, str(img_idx).zfill(5) + ".jpg")
    with tarfile.open(tar_path, 'r') as tar:
        jpg_names = sorted([name for name in tar.getnames() if name.endswith('.jpg')])
        assert jpg_names[img_idx-1] == imagename
        fileobj = tar.extractfile(imagename)
        image_data = fileobj.read()
        image_tmp = Image.open(BytesIO(image_data))
        image = copy.deepcopy(image_tmp)
        image_tmp.close()
        fileobj.close()
    return image


def extract_midframe_from_video(decord_vr, num_frames=32, start = None, end = None):
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    else:
        frame_id_list = frame_sample(duration, num_frames, mode=sample_scheme, local_fps=local_fps)
    # Extract the middle frame of 32 frames
    frame_id_list = [frame_id_list[15]]

    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()    
    images = [Image.fromarray(f.numpy()) if isinstance(f, torch.Tensor) else Image.fromarray(f) for f in video_data]
    return images


def eval(args, dataloader, model):
    df = pd.read_csv(args.anno_path)
    new_df = None
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_tensor = batch["image_tensor"].cuda()    
        with torch.no_grad():
            outputs = torch.softmax(model(image_tensor, inference=True), 1)
            preds = torch.argmax(outputs, 1)
        
        # save to a new df
        row_idx = batch["row_idx"]
        anno_idx = batch["anno_idx"]
        for i_ in range(preds.shape[0]):
            new_row = df.iloc[int(row_idx[i_])]
            new_row["shot_scale"] = preds[i_].item()
            if new_df is None:
                new_df = pd.DataFrame([new_row])
            else:
                new_df = pd.concat([new_df, pd.DataFrame([new_row])])
            
    if args.save_path:
        new_df.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help="Try to reduce batch size if OOM")  
    parser.add_argument('--dinov2_ckpt_path', type=str, default="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")
    parser.add_argument('--dinov2_config_path', type=str, default="dinov2/dinov2/configs/eval/vitb14_pretrain.yaml")

    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--anno_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--resume_path', type=str, default=None, help="path to shot scale classifier checkpoint")
    args = parser.parse_args()

    if args.save_path is None:
        print('please specify a saving path')
        sys.exit(0)

    # Load default DINOv2 models and parameters
    args_parser_dinov2 = get_args_parser_dinov2()
    args_dinov2 = args_parser_dinov2.parse_args()
    args_dinov2.pretrained_weights = args.dinov2_ckpt_path
    args_dinov2.config_file = args.dinov2_config_path
    dinov2_model, autocast_dtype = setup_and_build_model(args_dinov2)

    # Initialise shot scale classifer and load additional ckpts (for last 6 dinov2 layers and a linear layer)
    model = ScaleClassifier(dinov2_model, autocast_dtype).cuda()
    if args.resume_path:
        print('resuming from checkpoint')
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        print('no checkpoint found')
        sys.exit(0)


    val_dataset = ScaleDataset(dataset=args.dataset, video_root=args.video_dir, anno_df=args.anno_path)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    eval(args, val_dataloader, model)
    

