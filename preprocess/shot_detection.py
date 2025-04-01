import os
import sys
import copy
import ffmpeg
import decord
import tarfile
import argparse
import pandas as pd
import numpy as np

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from decord import VideoReader, cpu
from scenedetect import detect, ContentDetector, split_video_ffmpeg, AdaptiveDetector, SceneManager, FrameTimecode, open_video


def read_all_tarfile(tar_path): # Extract all frames from a tar file as PIL images
    images = []
    dirname = os.path.basename(tar_path.replace(".tar", ""))
    with tarfile.open(tar_path, 'r') as tar:
        members = sorted(tar.getmembers(), key=lambda member: member.name)
        for member in members:
            if member.isfile() and (member.name.endswith(".jpg") or member.name.endswith(".jpeg")):
                fileobj = tar.extractfile(member)
                if fileobj is not None:
                    image_data = fileobj.read()
                    image_tmp = Image.open(BytesIO(image_data))
                    image = copy.deepcopy(image_tmp)  
                    images.append(image)
                    image_tmp.close()
                    fileobj.close()     
    return images


def timecode_to_seconds(timecode):
    hours, minutes, seconds_milliseconds = timecode.split(":")
    seconds, milliseconds = seconds_milliseconds.split(".")
    total_seconds = int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    return total_seconds



def shot_detection(video_path, threshold=None): # Shot detection for videos (CMD-AD, MAD-Eval)
    if threshold is None: # CMD-AD
        scene_list = detect(video_path, AdaptiveDetector())
    else: # MAD-Eval
        scene_list = detect(video_path, AdaptiveDetector(adaptive_threshold=threshold)) 
    shot_list = []
    for scene in scene_list:
        shot_list.append((timecode_to_seconds(scene[0].get_timecode()), timecode_to_seconds(scene[1].get_timecode()))) 
    return shot_list


def shot_detection_from_images(images): # Shot detection for images (TV-AD)
    detector = AdaptiveDetector(adaptive_threshold= 1.5, min_scene_len=2, window_width=1)
    shot_list = []
    previous_cut = 1  # Start the first shot at frame 0
    frame_number = 0
    for image in images:
        frame_array = np.array(image)
        scene_cut = detector.process_frame(frame_number, frame_array)
        if scene_cut:
            # Append the interval from the previous cut to the current cut
            shot_list.append((previous_cut, frame_number - 1))
            previous_cut = frame_number  # Update the start of the next shot
        frame_number += 1
    # Append the final shot interval
    shot_list.append((previous_cut, (frame_number)))
    return shot_list




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--video_dir', default=None, type=str)
    parser.add_argument('--anno_path', default=None, type=str)
    parser.add_argument('--save_path', default=None, type=str)
    args = parser.parse_args()

    video_dir = args.video_dir
    anno_path = args.anno_path
    save_path = args.save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    '''
    Input: videos / a list of images
    Output: list of shots (with start and end time)
    '''

    if args.dataset == "cmdad":
        anno_df = pd.read_csv(anno_path)
        movie_clips = anno_df['cmd_filename'].unique()  
        
        movie_names = []
        imdbids = []
        cmd_filenames = []
        movie_titles = []
        shot_lists = []
        for movie_clip in tqdm(movie_clips, total=len(movie_clips)):
            single_anno_df = anno_df[anno_df['cmd_filename'] == movie_clip].iloc[0].copy()
            cmd_filename = single_anno_df['cmd_filename']
            video_path = os.path.join(video_dir, cmd_filename + '.mkv')

            # Save relevant information
            movie_names.append(single_anno_df['movie'])
            imdbids.append(single_anno_df['imdbid'])
            cmd_filenames.append(single_anno_df['cmd_filename'])
            movie_titles.append(single_anno_df['movie_title'])
            
            # Run shot detection function
            shot_list = shot_detection(video_path)
            
            # Add 20ms gap
            gapped_shot_list = []
            for shot_single in shot_list:
                gapped_shot_list.append((shot_single[0] + 0.01, shot_single[1] - 0.01))
            shot_lists.append(gapped_shot_list)
        
        # Saving
        output_df = pd.DataFrame.from_records({'movie':movie_names, 'imdbid':imdbids, 'cmd_filename':cmd_filenames, 'movie_title':movie_titles, 'shot_list':shot_lists})
        output_df.to_csv(save_path)

    
    elif args.dataset == "madeval":
        anno_df = pd.read_csv(anno_path)
        imdbids = anno_df["imdbid"].unique().tolist()
        
        video_paths = []
        shot_lists = []
        for imdbid in tqdm(imdbids, total=len(imdbids)):
            video_path = os.path.join(video_dir, imdbid + '.mkv')
            video_paths.append(video_path)

            # Run shot detection function
            shot_list = shot_detection(video_path, threshold=1.5)
            
            # Add 20ms gap
            gapped_shot_list = []
            for shot_single in shot_list:
                gapped_shot_list.append((shot_single[0] + 0.01, shot_single[1] - 0.01))
            shot_lists.append(gapped_shot_list)
        
        # Saving
        output_df = pd.DataFrame.from_records({'video_path':video_paths, 'imdbid':imdbids, 'shot_list':shot_lists})
        output_df.to_csv(save_path)
    
   
    elif args.dataset == "tvad":
        anno_df = pd.read_csv(anno_path)
        tv_clips = anno_df["tvad_name"].unique().tolist()
        
        video_filenames = []
        shot_lists = []
        success = []
        for clip in tqdm(tv_clips, total=len(tv_clips)):
            seg_name = clip
            # Determine if the clip is from bbt or friends
            if "friends" in seg_name:
                seg_path = os.path.join(video_dir, "friends_frames", seg_name + ".tar")
            else:
                seg_path = os.path.join(video_dir, "bbt_frames", seg_name + ".tar")
            video_filenames.append(seg_path)
            images = read_all_tarfile(seg_path)

            # Run shot detection function (for images input)
            shot_list = shot_detection_from_images(images)
            shot_lists.append(shot_list)

        # Saving
        output_df = pd.DataFrame.from_records({'video_filename':video_filenames, 'tvad_name':tv_clips, 'shot_list':shot_lists})
        output_df.to_csv(save_path)


    else:
        print("Please specify dataset")
        sys.exit(0)
