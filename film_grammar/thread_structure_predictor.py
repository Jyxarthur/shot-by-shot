import os
import cv2
import sys
import ast
import copy
import torch
import einops
import tarfile
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn.functional as F


from PIL import Image
from tqdm import tqdm
from io import BytesIO
from decord import VideoReader, cpu
from argparse import ArgumentParser
pd.options.mode.chained_assignment = None  

sys.path.append("dinov2")
from extract_feature import DINOv2_feature_extractor


def extract_frame_from_tarfile(tar_path, img_idx):
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


def extract_sideframes_from_video(decord_vr, num_frames=32, start = None, end = None):
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    else:
        frame_id_list = frame_sample(duration, num_frames, mode=sample_scheme, local_fps=local_fps)
    frame_id_list = [frame_id_list[0]] + [frame_id_list[-1]]
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()    
    images = [Image.fromarray(f.numpy()) if isinstance(f, torch.Tensor) else Image.fromarray(f) for f in video_data]
    return images


def softmax_topk(cos_sim, temperature, topk):
    cos_sim = (cos_sim / temperature).softmax(-1)
    tk_val, _ = torch.topk(cos_sim, dim=-1, k=topk)
    return tk_val.mean(-1).mean(-1)


def restrict_neighborhood(h, w, size_mask_neighborhood):
    mask = torch.zeros(h, w, h, w)
    for i in range(h):
        for j in range(w):
            for p in range(2 * size_mask_neighborhood + 1):
                for q in range(2 * size_mask_neighborhood + 1):
                    if i - size_mask_neighborhood + p < 0 or i - size_mask_neighborhood + p >= h:
                        continue
                    if j - size_mask_neighborhood + q < 0 or j - size_mask_neighborhood + q >= w:
                        continue
                    mask[i, j, i - size_mask_neighborhood + p, j - size_mask_neighborhood + q] = 1
    mask = mask.reshape(h * w, h * w)
    return mask.cuda(non_blocking=True)
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=1)
    parser.add_argument('--size_mask_neighborhood', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=0.1)

    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--anno_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()

    # Load DINOv2 model
    model_type = "vitg14_reg4"    
    model = DINOv2_feature_extractor(model_type, aspect_ratio=3/4)

    
    # Set attention mask (12 x 16 is the spatial feature size)
    neighbor_mask = restrict_neighborhood(h=12, w=16, size_mask_neighborhood=args.size_mask_neighborhood)
    
    df = pd.read_csv(args.anno_path)
    new_df = pd.DataFrame(columns=df.columns)
    new_df["thread"] = None
    last_video = ""
    anno_indices = sorted(df["anno_idx"].unique().tolist())
    for idx_, anno_idx in tqdm(enumerate(anno_indices), total=len(anno_indices)):
        sub_df = df[df["anno_idx"] == anno_idx]

        # Load images at start & end of each shot
        if args.dataset == "cmdad":
            cmd_filename = sub_df['cmd_filename'].iloc[0]
            video_path = os.path.join(args.video_dir, cmd_filename + '.mkv')  
            if video_path != last_video:
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
                last_video = video_path
            shot_images = []
            for idx, row in sub_df.iterrows():
                images = extract_sideframes_from_video(decord_vr, start = row['scaled_start'], end = row['scaled_end'])
                shot_images.extend(images)

        elif args.dataset == "madeval":
            imdbid = sub_df.iloc[0]["imdbid"]
            video_path = os.path.join(args.video_dir, imdbid + '.mkv')
            if video_path != last_video:
                decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
                last_video = video_path
            shot_images = []
            for idx, row in sub_df.iterrows():
                images = extract_sideframes_from_video(decord_vr, start = row['scaled_start'], end = row['scaled_end'])
                shot_images.extend(images)

        elif args.dataset == "tvad":
            seg_name = sub_df.iloc[0]["tvad_name"]
            if "friends" in seg_name:
                video_path = os.path.join(args.video_dir, "friends_frames", seg_name + ".tar")
            else:
                video_path = os.path.join(args.video_dir, "bbt_frames", seg_name + ".tar")
            shot_images = []
            for idx, row in sub_df.iterrows():
                start_idx = ast.literal_eval(row["scaled_index"])[0]
                end_idx = ast.literal_eval(row["scaled_index"])[-1]
                image_start = extract_frame_from_tarfile(video_path, start_idx)
                image_end = extract_frame_from_tarfile(video_path, end_idx)
                shot_images.append(image_start)
                shot_images.append(image_end)

        if len(sub_df) == 1: # only one shot
            clusters = [[0]]
            sub_df["thread"] = str(clusters)
        else:
            # Extract DINOv2 features
            features = model.extract_dino_v2_feat_from_pil(shot_images).detach().cpu().float()

            # Find the feature pairs for temporally nearest frames
            pairs = [(i, j) for i in range(len(sub_df)) for j in range(i + 1, len(sub_df))]
            features_start = features[0::2]
            features_end = features[1::2]
            feature_pairs = []
            for pair in pairs:
                feature_pairs.append(torch.stack([features_end[pair[0]], features_start[pair[1]]], 0))
            feature_pairs = torch.stack(feature_pairs, 0) 
            feature_pairs = F.normalize(feature_pairs, p=2, dim=-1)  # B 2 (hw + 1) c
            features_left = feature_pairs[:, 0, 1:].cuda() # B hw c
            features_right = feature_pairs[:, 1, 1:].cuda() # B hw c

            # Evaluate cosine similarity and apply softmax
            cos_sim = torch.bmm(features_left, features_right.transpose(1,2))
            cos_sim = (cos_sim * neighbor_mask[None, :, :])
            cos_sim_left = softmax_topk(cos_sim, temperature=args.temperature, topk=args.topk)
            cos_sim_right = softmax_topk(cos_sim.transpose(-1, -2), temperature=args.temperature, topk=args.topk)
            cos_sim_comb = (cos_sim_left + cos_sim_right) / 2

            # Thresholding and construct adjaceny matrix
            thres_pred = (cos_sim_comb.detach().cpu().numpy() > 0.3).astype(np.float32).tolist()

            adjacency = np.eye(len(sub_df))
            for pair_idx, pair in enumerate(pairs):
                if thres_pred[pair_idx] == 1:
                    adjacency[pair[0], pair[1]] = 1
                    adjacency[pair[1], pair[0]] = 1

            # Find connected components (clusters)
            G = nx.from_numpy_array(adjacency)
            clusters = list(nx.connected_components(G))
            clusters = [sorted(list(e)) for e in clusters]

            # Write to df
            sub_df["thread"] = str(clusters)

        if new_df is None:
            new_df = sub_df
        else:
            new_df = pd.concat([new_df, sub_df])

    new_df = new_df.reset_index(drop=True)
    if args.save_path:
        new_df.to_csv(args.save_path, index=False)

    