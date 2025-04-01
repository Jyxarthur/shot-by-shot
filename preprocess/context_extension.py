import os
import ast
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def find_overlapping_intervals(given_interval, interval_list):  # Output all indices in a list (shot lists) that overlap with the given interval (AD interval)
    a, b = given_interval
    overlapping_indices = []
    for idx, (start, end) in enumerate(interval_list):
        if max(a, start) <= min(b, end):
            overlapping_indices.append(idx)
    return overlapping_indices


def structure_context_by_shots(ad_start, ad_end, shot_list):
    # Find current shots (shots that overlap with AD interval)
    center_shot_indices = find_overlapping_intervals((ad_start, ad_end), shot_list)
    
    # "shot_times": the start and end times for past, current, future shots
    # "valid_shots": specify whether the corresponding shot is valid
    if len(center_shot_indices) == 0:
        shot_times = np.zeros((5, 2))
        shot_times[2, 0] = ad_start
        shot_times[2, 1] = ad_end
        valid_shots = np.array([0., 0., 1., 0., 0.])
    else:
        shot_indices = [center_shot_indices[0] - 2, center_shot_indices[0] - 1] + center_shot_indices + [center_shot_indices[-1] + 1, center_shot_indices[-1] + 2]
        max_time = shot_list[-1][1] # The duration of video
        
        # Check if all shots are within the video duration
        shot_times_raw = []
        valid_shots_raw = []
        for shot_idx in shot_indices:
            if shot_idx < 0 or shot_idx >= len(shot_list):
                valid_shots_raw.append(0)
                shot_times_raw.append((-1, -1))
            else:
                valid_shots_raw.append(1)
                shot_times_raw.append((round(shot_list[shot_idx][0], 3), round(shot_list[shot_idx][1], 3)))
        shot_times_raw = np.array(shot_times_raw)

        # Specify a set of boundaries: 
        # Current shot max range: [AD_interval-internal_range, AD_interval+internal_range]; 
        # Past & future shot max range: [AD_interval-internal_range-external_range*2, AD_interval+internal_range+external_range*2]
        external = args.external_range
        internal = args.internal_range
        if len(center_shot_indices) == 1:
            center_thresholds = [(ad_start - internal, ad_end + internal)]
        else:
            center_thresholds = [(ad_start - internal, max_time)] + (len(center_shot_indices) - 2) * [(0, max_time)] + [(0, ad_end + internal)]
        max_or_min = np.array([(1, 1)] * 2 + (len(center_shot_indices)) * [(1, 0)] + [(0, 0)] * 2)
        thresholds = [(ad_start - external * 2 - internal, 0), (ad_start - external * 2 - internal, 0)] + \
                    center_thresholds + [(max_time, ad_end + external * 2 + internal), (max_time, ad_end + external * 2 + internal)]
        thresholds = np.array(thresholds)
        lower_bound = np.zeros_like(thresholds)
        upper_bound = np.zeros_like(thresholds) + max_time
        shot_times_max = np.max(np.stack([shot_times_raw, thresholds, lower_bound], 0), 0)
        shot_times_min = np.min(np.stack([shot_times_raw, thresholds, upper_bound], 0), 0)
        shot_times = shot_times_max * max_or_min + shot_times_min * (1 - max_or_min)
        # Remove shots that are too short (< 0.4s)
        valid_shots = ((shot_times[:,1] - shot_times[:,0]) > 0.4).astype(np.float32)
    return shot_times, valid_shots


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--anno_path', default=None, type=str)
    parser.add_argument('--shot_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--internal_range', default=3., type=float, help='The maximum extension of current shots from AD intervals')
    parser.add_argument('--external_range', default=8., type=float, help='The maximum duration of each past or future shots')
    args = parser.parse_args()

    anno_path = args.anno_path
    shot_path = args.shot_path
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    '''
    Input: list of shots; AD interval
    Output: Shot structures (including context shots) for all AD clips
    '''

    args.dataset = "tvad"

    if args.dataset == "tvad": # multiple the range in second to obtain the range in frames (for 3fps images)
        args.internal_range = int(args.internal_range * 3)
        args.external_range = int(args.external_range * 3)

    anno_df = pd.read_csv(anno_path)
    shot_df = pd.read_csv(shot_path)  
    anno_df_shots = pd.DataFrame(columns=anno_df.columns)

    for anno_idx_per_ad, (anno_df_row_idx, anno_df_row) in tqdm(enumerate(anno_df.iterrows()), total = len(anno_df)):
        # Extract AD interval: retrieve back the AD start and end time in video (rather than AD duration "scaled" to at least 2s)
        if args.dataset == "cmdad":
            ad_mid = (anno_df_row["start"] + anno_df_row["end"]) / 2
            scaled_mid = (anno_df_row["scaled_start"] + anno_df_row["scaled_end"]) / 2
            ad_duration = anno_df_row["end"] - anno_df_row["start"]
            ad_start = max(scaled_mid - ad_duration / 2, 0)
            ad_end = max(scaled_mid + ad_duration / 2, 0)
            shot_df_single = shot_df[shot_df["cmd_filename"] == anno_df_row["cmd_filename"]]
        elif args.dataset == "madeval":
            ad_start = anno_df_row["start"]
            ad_end = anno_df_row["end"]
            shot_df_single = shot_df[shot_df["imdbid"] == anno_df_row["imdbid"]]
        elif args.dataset == "tvad":
            ad_start = min(ast.literal_eval(anno_df_row["tvad_index"]))
            ad_end = max(ast.literal_eval(anno_df_row["tvad_index"]))
            shot_df_single = shot_df[shot_df["tvad_name"] == anno_df_row["tvad_name"]]
        else:
            print("Please specify the dataset")

        # Extract shot list
        shot_list = ast.literal_eval(shot_df_single["shot_list"].iloc[0])

        # Find start & end for context shots ("shot_times"), and whether the context shot is valid ("valid_shots")
        shot_times, valid_shots = structure_context_by_shots(ad_start, ad_end, shot_list) 

        # Add shot labels ("shot_label"): past-l (left), future-r (right), current-m (middle)
        for shot_idx, shot_time in enumerate(shot_times):
            if valid_shots[shot_idx] == 0.:
                continue
            anno_df_shot_row = anno_df_row.copy()
            
            if args.dataset == "tvad":
                anno_df_shot_row["scaled_index"] = str(list(range(shot_time[0], shot_time[1]+1)))
                anno_df_shot_row["AD_index"] = str(list(range(ad_start, ad_end+1)))
                anno_df_shot_row["AD_start"] = anno_df_shot_row["start"]
                anno_df_shot_row["AD_end"] = anno_df_shot_row["end"]
            else:
                anno_df_shot_row["scaled_start"] = round(shot_time[0], 3)
                anno_df_shot_row["scaled_end"] = round(shot_time[1], 3)
                anno_df_shot_row["AD_start"] = round(ad_start, 3)
                anno_df_shot_row["AD_end"] = round(ad_end, 3)

            if shot_idx == 0:
                anno_df_shot_row["shot_label"] = "l-" + "0"
            elif shot_idx == 1:
                anno_df_shot_row["shot_label"] = "l-" + "1"
            elif shot_idx == len(shot_times) - 2:
                anno_df_shot_row["shot_label"] = "r-" + "0"
            elif shot_idx == len(shot_times) - 1:
                anno_df_shot_row["shot_label"] = "r-" + "1"
            else:
                anno_df_shot_row["shot_label"] = "m-" + str(shot_idx - 2)
            anno_df_shot_row["anno_idx"] = int(anno_idx_per_ad)   # The index of AD in the dataset
            anno_df_shots = pd.concat([anno_df_shots, pd.DataFrame([anno_df_shot_row])], ignore_index=True)

    anno_df_shots.to_csv(save_path, index=False)

    
