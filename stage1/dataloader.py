import os
import sys
import ast
import copy
import json
import torch
import pickle
import tarfile
import numpy as np
import pandas as pd
import tarfile
from io import BytesIO
from decord import VideoReader, cpu
import torchvision.transforms as transforms
import pandas as pd
pd.options.mode.chained_assignment = None  # Disable the warning

from PIL import Image, ImageDraw, ImageFont
from utils import process_video, partition_array, get_context_timearray, uniform_sampling, \
                group_values, extract_ordered_keys, read_tarfile, convert_bounding_box_to_rectangle, \
                convert_bounding_box_to_ellipse, process_text_prompt, process_video_selected_idx, \
                process_video_selected_idx_cache, fetch_video_from_tensor, pil_to_base64_jpg

class CMDAD_Dataset():
    def __init__(self,
                model,
                processor,
                general_prompt,
                label_type, 
                label_width, 
                label_alpha,
                shot_label,
                adframe_label,
                anno_path,
                video_dir,
                charbank_path,
                font_path,
                num_frames,
                prompt_idx,
                **kwargs):

        self.model = model
        self.processor = processor
        self.prompt_idx = prompt_idx
        self.num_frames = num_frames
        self.transform = transforms.Resize((336,))
        self.general_prompt=general_prompt
        self.font_path = font_path

        # Label information, including colour coding, type, visual prompts, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha
        self.shot_label=shot_label
        self.adframe_label=adframe_label

        # Load source file
        self.anno_df = pd.read_csv(anno_path)

        # Prepare character bank as dictionaries {name_id: role}
        self.charbank_dict = {}
        with open(os.path.join(charbank_path)) as fobj:
            charbank_dict = json.load(fobj)
        for key in charbank_dict.keys():
            self.charbank_dict[key] = {single_charbank_dict["id"]:single_charbank_dict["role"] for single_charbank_dict in charbank_dict[key]}    

        self.all_clips = []
        anno_indices = self.anno_df["anno_idx"].unique().tolist()
        for anno_idx in anno_indices:
            clip_df = self.anno_df[self.anno_df["anno_idx"] == anno_idx]
            imdbid = clip_df.iloc[0]["imdbid"]
            text_gt = clip_df.iloc[0]['text']

            # The AD interval
            gt_start = clip_df["AD_start"].iloc[0]
            gt_end = clip_df["AD_end"].iloc[0]

            # Video path
            cmd_filename = clip_df.iloc[0]['cmd_filename']
            video_path = os.path.join(video_dir, cmd_filename + '.mkv')
            if not os.path.exists(video_path):
                continue
            
            # Predicted thread structures
            threads = ast.literal_eval(clip_df.iloc[0]["thread"])
            
            # Per-shot information
            shot_starts = []
            shot_ends = []
            shot_bboxes = []
            shot_pred_ids = []
            shot_labels = []
            shot_scales = []
            for shot_idx, shot_row in clip_df.iterrows():
                shot_starts.append(shot_row["scaled_start"])
                shot_ends.append(shot_row["scaled_end"])
                shot_bboxes.append(shot_row["bboxes"])
                shot_pred_ids.append(shot_row["pred_ids"])
                shot_labels.append(shot_row["shot_label"])
                shot_scales.append(shot_row["shot_scale"])
            self.all_clips.append([imdbid, anno_idx, cmd_filename, video_path, threads, text_gt, gt_start, gt_end, shot_starts, shot_ends, shot_bboxes, shot_pred_ids, shot_labels, shot_scales])
        print(f"In total {len(self.all_clips)} CMD-AD clips")

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        imdbid, anno_idx, cmd_filename, video_path, threads, text_gt, gt_start, gt_end, shot_starts, shot_ends, shot_bboxes, shot_pred_ids, shot_labels, shot_scales = self.all_clips[index]
        
        '''
        Frame indices sampling: 16 frames AD interval + 16 frames context frames
        '''
        shot_boundary = [e + 0.01 for e in shot_ends[:-1]]
        current_shots = [j for j, e in enumerate(shot_labels) if "m" in e]  # indices of current shots
        gt_start = max(gt_start, shot_starts[current_shots[0]])
        gt_end = min(gt_end, shot_ends[current_shots[-1]])
        gt_duration = gt_end - gt_start

        # For current shot --- || left current frames | gt (AD interval, 16 frames) | right current frames ||; left current frames + right current frames are at most (self.num_frames // 2 = 8) frames 
        left_current_duration = gt_start - shot_starts[current_shots[0]]
        right_current_duration = shot_ends[current_shots[-1]] - gt_end
        current_frames = min(round(((left_current_duration + right_current_duration) / (2 * gt_duration + 1e-8) * self.num_frames) // 2) * 2, int(self.num_frames / 2)) # this excludes AD interval (16 frames)
        left_current_frames = round(current_frames * left_current_duration / (left_current_duration + right_current_duration + 1e-8))
        right_current_frames = current_frames - left_current_frames

        # Find timestamps in current shots 
        ad_timearray = np.linspace(gt_start, gt_end, self.num_frames, endpoint=False)
        if left_current_frames != 0:
            left_current_array = np.linspace(shot_starts[current_shots[0]], gt_start, left_current_frames, endpoint=False) 
        else:
            left_current_array = np.array([])
        if right_current_frames != 0:
            right_current_array = np.linspace(gt_end, shot_ends[current_shots[-1]], right_current_frames, endpoint=False) 
        else:
            right_current_array = np.array([])
        current_timearray = left_current_array.tolist() + ad_timearray.tolist() + right_current_array.tolist()
        current_split_timearray = partition_array(np.array(current_timearray), shot_boundary)

        # Find timestamps in past and future frames 
        past_shots = [j for j, e in enumerate(shot_labels) if "l" in e]
        future_shots = [j for j, e in enumerate(shot_labels) if "r" in e]
        past_split_timearray, future_split_timearray = get_context_timearray(self.num_frames - current_frames, past_shots, future_shots, shot_starts, shot_ends, shot_boundary)

        if len(past_split_timearray) == 0 and len(future_split_timearray) != 0:
            split_timearray = [np.concatenate([j, k], 0) for j, k in zip(current_split_timearray, future_split_timearray)]
        elif len(past_split_timearray) != 0 and len(future_split_timearray) == 0:
            split_timearray = [np.concatenate([j, k], 0) for j, k in zip(past_split_timearray, current_split_timearray)]
        elif len(past_split_timearray) != 0 and len(future_split_timearray) != 0:
            split_timearray = [np.concatenate([i, j, k], 0) for i, j, k in zip(past_split_timearray, current_split_timearray, future_split_timearray)]
        else:
            split_timearray = current_split_timearray
        
        '''
        Read frames (with visual prompts)
        '''
        cache = None  # decord cache 
        all_frames = [] # all processed frames
        all_name_ids = {} # all character information
        shot_labels_refined = []
        shot_scales_refined = []
        shot_idx_tmp = 0
        valid_shots = {}
        for shot_idx, start in enumerate(shot_starts):
            split_timearray_single = split_timearray[shot_idx]
            if len(split_timearray_single) == 0:
                all_frames.extend([])
                valid_shots[shot_idx] = -1
                continue
            end = shot_ends[shot_idx]

            # Since character recognition are performed with 32 frames for each shot, we need to find the closest character-recognised frame for each timestamp
            shot_timearray = np.linspace(start, end, self.num_frames * 2, endpoint=False)
            selected_idx = [np.argmin(np.abs(shot_timearray - b)) for b in split_timearray_single]
            selected_timearray = shot_timearray[selected_idx]

            # Read frames
            bboxes = shot_bboxes[shot_idx]
            pred_ids = shot_pred_ids[shot_idx]
            frames, cache, all_name_ids, processed_frames = self.read_and_label_shot(shot_idx_tmp, gt_start, gt_end, start, end, selected_idx, selected_timearray, video_path, bboxes, pred_ids, all_name_ids, cache = cache)
            all_frames.extend(processed_frames)

            # Since some shots are too short in time, which are not sampled at all, we need to correct the shot scales and thread structures accordingly
            shot_labels_refined.append(shot_labels[shot_idx])
            shot_scales_refined.append(shot_scales[shot_idx])
            valid_shots[shot_idx] = shot_idx_tmp
            shot_idx_tmp += 1


        # Refine all shot information (due to occational unsampled shots)
        current_shots_refined = [j for j, e in enumerate(shot_labels_refined) if "m" in e]
        threads = [e for e in threads if len(e) > 1]  # only consider thread with more than one shot
        threads_refined = []
        for thread in threads:
            thread_refined = []
            for e in thread:
                if valid_shots[e] != -1:
                    thread_refined.append(valid_shots[e])
            if len(thread_refined) > 1:
                threads_refined.append(thread_refined)

        if self.model == "qwen2vl":
            # Convert to tensors
            all_frames = np.stack(all_frames, 0)
            video_tensor = torch.from_numpy(all_frames).permute(0, 3, 1, 2)
            video = fetch_video_from_tensor(video_tensor) 
        elif self.model == "gpt4o":  
            # Convert to base64 images    
            base64_images = []
            for frame_idx, processed_frame in enumerate(all_frames):
                base64_images.append(pil_to_base64_jpg(self.transform(processed_frame)))
            video = base64_images
        else:
            print("please specify 'qwen2vl' or 'gpt4o' as the model")
            sys.exit()
        
        '''
        Formulate character information and text prompts
        '''
        charbank_dict = self.charbank_dict[imdbid]
        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending  #.split(" ")[0]
        if char_text == ". Possible characters (labeled by {label_type}): " or self.label_type == "none": 
            char_text = ""
        else:
            char_text=char_text.format(label_type=self.label_type)

        if self.model == "qwen2vl":
            processed_text_prompt, text_prompt = process_text_prompt(self.general_prompt.apply(char_text, current_shots=current_shots_refined, threads=threads_refined, shot_scales=shot_scales_refined), self.processor)
        elif self.model == "gpt4o": 
            text_prompt = self.general_prompt.apply(char_text, current_shots=current_shots_refined, threads=threads_refined, shot_scales=shot_scales_refined)
            processed_text_prompt = text_prompt
        else:
            print("please specify 'qwen2vl' or 'gpt4o' as the model")
            sys.exit()

        return_dict =  {'video': video, \
                'imdbid': imdbid, \
                'processed_prompt': processed_text_prompt,
                'prompt': text_prompt,
                'gt_text': text_gt,
                'start': gt_start,  # AD interval start
                'end': gt_end,  # AD interval end
                'start_': shot_starts[0], # whole clip (AD interval + context) start
                'end_': shot_ends[-1], # whole clip (AD interval + context) end
                'anno_idx': anno_idx,
                }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['processed_prompt'] = [sample['processed_prompt'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        out_batch['anno_idx'] = [sample['anno_idx'] for sample in batch]
        return out_batch

    def read_and_label_shot(self, shot_idx, gt_start, gt_end, start, end, selected_idx, selected_timearray, video_path, bboxes, name_ids, all_name_ids, cache = None):
        '''
        Read selected frames for each shot
        Add colour-coded circles/boxes around character faces
        Add shot labels
        (Optional) Add outline box for AD interval frames
        '''
        # Read frames and add character face labels
        frames, decord_vr = process_video_selected_idx(video_path, selected_idx, start=start, end=end, cache = None)
        if self.label_type=="none":
            processed_frames = frames
        else:
            bboxes = ast.literal_eval(bboxes)
            name_ids = ast.literal_eval(name_ids)   # {"ttxxxxxx": 1, ...]
            named_bboxes = [{} for i in range(len(bboxes))]   # [{"ttxxxxxx": [bbox], ...}, {}, {}, ...]
            for name_id, bbox_info in name_ids.items():
                for bbox_info_single in bbox_info:
                    named_bboxes[bbox_info_single[0]][name_id] = bboxes[bbox_info_single[0]][bbox_info_single[1]]

            bboxes_filtered = np.array(named_bboxes, dtype=object)[selected_idx].tolist()
            shot_name_ids = extract_ordered_keys(bboxes_filtered)
            for shot_name_id in shot_name_ids:
                if shot_name_id not in all_name_ids.keys():
                    all_name_ids[shot_name_id] = len(all_name_ids)

            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: 
                    processed_frames.append(frame)
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # Draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()
                        # Overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # Overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))            
                    processed_frames.append(processed_frame)
        
        # Add shot numbers
        if self.shot_label:
            final_frames = []
            for frame_idx, processed_frame in enumerate(processed_frames):
                final_frames.append(self.draw_shot_label(shot_idx, processed_frame))
        else:
            final_frames = processed_frames

        # Add outline boxes
        if self.adframe_label:
            final_frames_ = []
            for frame_idx, processed_frame in enumerate(final_frames):
                selected_time = selected_timearray[frame_idx]
                if selected_time >= (gt_start - 0.01) and selected_time <= (gt_end + 0.01):
                    final_frames_.append(self.add_red_border(processed_frame, border_thickness=5))
                else:
                    final_frames_.append(processed_frame)
        else:
            final_frames_ = final_frames

        return frames, decord_vr, all_name_ids, final_frames_

    def draw_shot_label(self, shot_idx, image):
        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Set font (default to a simple font if PIL cannot find a specific one)
        font = ImageFont.truetype(self.font_path, 30)

        # Set text
        text = f"Shot {shot_idx}"

        # Calculate text size to draw a background rectangle
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Get bounding box of the text
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        padding = 5
        background_position = (0, 0, text_width + 2 * padding, text_height + 4 * padding)

        # Draw white rectangle as background for the text
        draw.rectangle(background_position, fill="white")

        # Draw the text in black on top of the white rectangle
        text_position = (padding, padding)
        draw.text(text_position, text, fill="black", font=font)

        return image

    def add_red_border(self, image, border_thickness=5):
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions
        width, height = image.size
        
        # Draw a red rectangle as the border
        for i in range(border_thickness):
            draw.rectangle(
                [i, i, width - i - 1, height - i - 1], outline="red"
            )
        
        return image



class MADEval_Dataset():
    def __init__(self,
                model, 
                processor,
                general_prompt,
                label_type, 
                label_width, 
                label_alpha,
                shot_label,
                adframe_label,
                anno_path,
                video_dir,
                charbank_path,
                font_path,
                num_frames,
                prompt_idx,
                **kwargs):
        
        self.model = model
        self.processor = processor
        self.general_prompt=general_prompt
        self.num_frames = num_frames
        self.transform = transforms.Resize((336,))
        self.prompt_idx = prompt_idx
        self.font_path = font_path

        # Label information, including colour coding, type, visual prompts, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha
        self.shot_label=shot_label
        self.adframe_label=adframe_label

        # Load source file
        self.anno_df = pd.read_csv(anno_path)

        # Prepare character bank as dictionaries {name_id: role}
        self.charbank_dict = {}
        charbank_pred = pickle.load(open(charbank_path,'rb'))
        charbank_pred = {str(key):value for key,value in charbank_pred.items()}
        for movie_name in charbank_pred.keys():
            charbank_pred_per_frame = charbank_pred[movie_name]
            name_ids_per_frame = charbank_pred_per_frame["charbank_nmids"].iloc[0]
            roles_per_frame = charbank_pred_per_frame["charbank_roles"].iloc[0]
            self.charbank_dict[movie_name] = {k: v for k, v in zip(name_ids_per_frame, roles_per_frame)}

        self.all_clips = []
        anno_indices = self.anno_df["anno_idx"].unique().tolist()
        for anno_idx in anno_indices:
            clip_df = self.anno_df[self.anno_df["anno_idx"] == anno_idx]
            text_gt = clip_df.iloc[0]['sentence']
            movie_name = clip_df.iloc[0]['movie']

            # The AD interval
            gt_start = clip_df["AD_start"].iloc[0]
            gt_end = clip_df["AD_end"].iloc[0]

            # Video path
            imdbid = clip_df.iloc[0]["imdbid"]
            video_path = os.path.join(video_dir, imdbid + '.mkv')
            if not os.path.exists(video_path):
                continue
            
            # Predicted thread structures
            threads = ast.literal_eval(clip_df.iloc[0]["thread"])                

            # Per-shot information
            shot_starts = []
            shot_ends = []
            shot_bboxes = []
            shot_pred_ids = []
            shot_labels = []
            shot_scales = []
            for shot_idx, shot_row in clip_df.iterrows():
                shot_starts.append(shot_row["scaled_start"])
                shot_ends.append(shot_row["scaled_end"])
                shot_bboxes.append(shot_row["bboxes"])
                shot_pred_ids.append(shot_row["pred_ids"])
                shot_labels.append(shot_row["shot_label"])
                shot_scales.append(shot_row["shot_scale"])
            self.all_clips.append([imdbid, anno_idx, movie_name, video_path, threads, text_gt, gt_start, gt_end, shot_starts, shot_ends, shot_bboxes, shot_pred_ids, shot_labels, shot_scales])
        print(f"In total {len(self.all_clips)} MADEval clips")

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, index):
        imdbid, anno_idx, movie_name, video_path, threads, text_gt, gt_start, gt_end, shot_starts, shot_ends, shot_bboxes, shot_pred_ids, shot_labels, shot_scales = self.all_clips[index]

        '''
        Frame indices sampling: 16 frames AD interval + 16 frames context frames
        '''
        shot_boundary = [e + 0.01 for e in shot_ends[:-1]]
        current_shots = [j for j, e in enumerate(shot_labels) if "m" in e] 
        gt_start = max(gt_start, shot_starts[current_shots[0]])
        gt_end = min(gt_end, shot_ends[current_shots[-1]])
        gt_duration = gt_end - gt_start

        # For current shot --- || left current frames | gt (AD interval, 16 frames) | right current frames ||; left current frames + right current frames are at most (self.num_frames // 2 = 8) frames 
        left_current_duration = gt_start - shot_starts[current_shots[0]]
        right_current_duration = shot_ends[current_shots[-1]] - gt_end
        current_frames = min(round(((left_current_duration + right_current_duration) / (2 * gt_duration + 1e-8) * self.num_frames) // 2) * 2, int(self.num_frames / 2)) # this excludes AD interval (16 frames)
        left_current_frames = round(current_frames * left_current_duration / (left_current_duration + right_current_duration + 1e-8))
        right_current_frames = current_frames - left_current_frames

        # Find timestamps in current shots 
        ad_timearray = np.linspace(gt_start, gt_end, self.num_frames, endpoint=False)
        if left_current_frames != 0:
            left_current_array = np.linspace(shot_starts[current_shots[0]], gt_start, left_current_frames, endpoint=False) 
        else:
            left_current_array = np.array([])
        if right_current_frames != 0:
            right_current_array = np.linspace(gt_end, shot_ends[current_shots[-1]], right_current_frames, endpoint=False) 
        else:
            right_current_array = np.array([])
        current_timearray = left_current_array.tolist() + ad_timearray.tolist() + right_current_array.tolist()
        current_split_timearray = partition_array(np.array(current_timearray), shot_boundary)

        # Find timestamps in past and future frames 
        past_shots = [j for j, e in enumerate(shot_labels) if "l" in e]
        future_shots = [j for j, e in enumerate(shot_labels) if "r" in e]
        past_split_timearray, future_split_timearray = get_context_timearray(self.num_frames - current_frames, past_shots, future_shots, shot_starts, shot_ends, shot_boundary)

        if len(past_split_timearray) == 0 and len(future_split_timearray) != 0:
            split_timearray = [np.concatenate([j, k], 0) for j, k in zip(current_split_timearray, future_split_timearray)]
        elif len(past_split_timearray) != 0 and len(future_split_timearray) == 0:
            split_timearray = [np.concatenate([j, k], 0) for j, k in zip(past_split_timearray, current_split_timearray)]
        elif len(past_split_timearray) != 0 and len(future_split_timearray) != 0:
            split_timearray = [np.concatenate([i, j, k], 0) for i, j, k in zip(past_split_timearray, current_split_timearray, future_split_timearray)]
        else:
            split_timearray = current_split_timearray
        
        '''
        Read frames (with visual prompts)
        '''
        cache = None  # decord cache 
        all_frames = [] # all processed frames
        all_name_ids = {} # all character information
        shot_labels_refined = []
        shot_scales_refined = []
        shot_idx_tmp = 0
        valid_shots = {}
        for shot_idx, start in enumerate(shot_starts):
            split_timearray_single = split_timearray[shot_idx]
            if len(split_timearray_single) == 0:
                all_frames.extend([])
                valid_shots[shot_idx] = -1
                continue
            end = shot_ends[shot_idx]

            # Since character recognition are performed with 32 frames for each shot, we need to find the closest character-recognised frame for each timestamp
            shot_timearray = np.linspace(start, end, self.num_frames * 2, endpoint=False)
            selected_idx = [np.argmin(np.abs(shot_timearray - b)) for b in split_timearray_single]
            selected_timearray = shot_timearray[selected_idx]

            # Read frames
            bboxes = shot_bboxes[shot_idx]
            bboxes = str([[[int(coord * 480 / 720) for coord in bbox] for bbox in group] for group in ast.literal_eval(bboxes)])
            pred_ids = shot_pred_ids[shot_idx]
            frames, cache, all_name_ids, processed_frames = self.read_and_label_shot(shot_idx_tmp, gt_start, gt_end, start, end, selected_idx, selected_timearray, video_path, bboxes, pred_ids, all_name_ids, cache = cache)
            all_frames.extend(processed_frames)

            # Since some shots are too short in time, which are not sampled at all, we need to correct the shot scales and thread structures accordingly
            shot_labels_refined.append(shot_labels[shot_idx])
            shot_scales_refined.append(shot_scales[shot_idx])
            valid_shots[shot_idx] = shot_idx_tmp
            shot_idx_tmp += 1

        # Refine all shot information (due to occational unsampled shots)
        current_shots_refined = [j for j, e in enumerate(shot_labels_refined) if "m" in e]
        threads = [e for e in threads if len(e) > 1]  # only consider thread with more than one shot
        threads_refined = []
        for thread in threads:
            thread_refined = []
            for e in thread:
                if valid_shots[e] != -1:
                    thread_refined.append(valid_shots[e])
            if len(thread_refined) > 1:
                threads_refined.append(thread_refined)

        if self.model == "qwen2vl":
            # Convert to tensors
            all_frames = np.stack(all_frames, 0)
            video_tensor = torch.from_numpy(all_frames).permute(0, 3, 1, 2)
            video = fetch_video_from_tensor(video_tensor) 
        elif self.model == "gpt4o":  
            # Convert to base64 images    
            base64_images = []
            for frame_idx, processed_frame in enumerate(all_frames):
                base64_images.append(pil_to_base64_jpg(self.transform(processed_frame)))
            video = base64_images
        else:
            print("please specify 'qwen2vl' or 'gpt4o' as the model")
            sys.exit()       

        '''
        Formulate character information and text prompts
        '''
        charbank_dict = self.charbank_dict[movie_name]
        char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            char_text = char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending  #.split(" ")[0]
        if char_text == ". Possible characters (labeled by {label_type}): " or self.label_type == "none": 
            char_text = ""
        else:
            char_text=char_text.format(label_type=self.label_type)

        if self.model == "qwen2vl":
            processed_text_prompt, text_prompt = process_text_prompt(self.general_prompt.apply(char_text, current_shots=current_shots_refined, threads=threads_refined, shot_scales=shot_scales_refined), self.processor)
        elif self.model == "gpt4o": 
            text_prompt = self.general_prompt.apply(char_text, current_shots=current_shots_refined, threads=threads_refined, shot_scales=shot_scales_refined)
            processed_text_prompt = text_prompt
        else:
            print("please specify 'qwen2vl' or 'gpt4o' as the model")
            sys.exit()

        return_dict =  {'video': video, \
                'imdbid': imdbid, \
                'processed_prompt': processed_text_prompt,
                'prompt': text_prompt,
                'gt_text': text_gt,
                'start': gt_start,  # AD interval start
                'end': gt_end,  # AD interval end
                'start_': shot_starts[0], # whole clip (AD interval + context) start
                'end_': shot_ends[-1], # whole clip (AD interval + context) end
                'anno_idx': anno_idx,
                }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['processed_prompt'] = [sample['processed_prompt'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        out_batch['anno_idx'] = [sample['anno_idx'] for sample in batch]
        return out_batch

    def read_and_label_shot(self, shot_idx, gt_start, gt_end, start, end, selected_idx, selected_timearray, video_path, bboxes, name_ids, all_name_ids, cache):
        '''
        Read selected frames for each shot
        Add colour-coded circles/boxes around character faces
        Add shot labels
        (Optional) Add outline box for AD interval frames
        '''
        # Read frames and add character face labels
        frames, decord_vr = process_video_selected_idx_cache(video_path, selected_idx, start=start, end=end, cache = cache)
        if self.label_type=="none":
            processed_frames = frames
        else:
            bboxes = ast.literal_eval(bboxes)   
            name_ids = ast.literal_eval(name_ids)   # {"ttxxxxxx": 1, ...]
            named_bboxes = [{} for i in range(len(bboxes))]  # [{"ttxxxxxx": [bbox], ...}, {}, {}, ...]
            for name_id, bbox_info in name_ids.items():
                for bbox_info_single in bbox_info:
                    named_bboxes[bbox_info_single[0]][name_id] = bboxes[bbox_info_single[0]][bbox_info_single[1]]

            bboxes_filtered = np.array(named_bboxes, dtype=object)[selected_idx].tolist()
            shot_name_ids = extract_ordered_keys(bboxes_filtered)
            for shot_name_id in shot_name_ids:
                if shot_name_id not in all_name_ids.keys():
                    all_name_ids[shot_name_id] = len(all_name_ids)

            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: 
                    processed_frames.append(frame)
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # Draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()
                        # Overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # Overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))            
                    processed_frames.append(processed_frame)
        
        # Add shot numbers
        if self.shot_label:
            final_frames = []
            for frame_idx, processed_frame in enumerate(processed_frames):
                final_frames.append(self.draw_shot_label(shot_idx, processed_frame))
        else:
            final_frames = processed_frames

         # Add outline boxes
        if self.adframe_label:
            final_frames_ = []
            for frame_idx, processed_frame in enumerate(final_frames):
                selected_time = selected_timearray[frame_idx]
                if selected_time >= (gt_start - 0.01) and selected_time <= (gt_end + 0.01):
                    final_frames_.append(self.add_red_border(processed_frame, border_thickness=7))
                else:
                    final_frames_.append(processed_frame)
        else:
            final_frames_ = final_frames

        return frames, decord_vr, all_name_ids, final_frames_

    def draw_shot_label(self, shot_idx, image):
        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Set font (default to a simple font if PIL cannot find a specific one)
        font = ImageFont.truetype(self.font_path, 40)

        # Set text
        text = f"Shot {shot_idx}"

        # Calculate text size to draw a background rectangle
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Get bounding box of the text
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        padding = 7
        background_position = (0, 0, text_width + 2 * padding, text_height + 4 * padding)

        # Draw white rectangle as background for the text
        draw.rectangle(background_position, fill="white")

        # Draw the text in black on top of the white rectangle
        text_position = (padding, padding)
        draw.text(text_position, text, fill="black", font=font)

        return image

    def add_red_border(self, image, border_thickness=5):
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions
        width, height = image.size
        
        # Draw a red rectangle as the border
        for i in range(border_thickness):
            draw.rectangle(
                [i, i, width - i - 1, height - i - 1], outline="red"
            )
        
        return image




class TVAD_Dataset():
    def __init__(self,
                model,
                processor,
                general_prompt, 
                label_type, 
                label_width, 
                label_alpha,
                shot_label,
                adframe_label,
                anno_path,
                video_dir,
                charbank_path,
                font_path,
                num_frames,
                prompt_idx,
                **kwargs):

        self.model = model
        self.processor = processor
        self.general_prompt = general_prompt
        self.num_frames = num_frames
        self.transform = transforms.Resize((336,))
        self.prompt_idx = prompt_idx
        self.font_path = font_path

        # Label information, including colour coding, type, visual prompts, etc.
        self.colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        self.color_name = ["red", "green", "blue", "yellow", "pink", "cyan", "white", "black", "black", "black", "black", "black"]
        self.label_type=label_type
        self.label_width=label_width
        self.label_alpha=label_alpha
        self.shot_label=shot_label
        self.adframe_label=adframe_label

        # Load source file
        self.anno_df = pd.read_csv(anno_path)

        # Prepare character bank as dictionaries {name_id: role}
        self.charbank_dict = {}
        with open(os.path.join(charbank_path)) as fobj:
            charbank_dict = json.load(fobj)
        for key in charbank_dict.keys():
            self.charbank_dict[key] = {single_charbank_dict["id"]:single_charbank_dict["role"] for single_charbank_dict in charbank_dict[key]}         

        self.all_clips = []
        anno_indices = self.anno_df["anno_idx"].unique().tolist()
        for anno_idx in anno_indices:
            clip_df = self.anno_df[self.anno_df["anno_idx"] == anno_idx]
            imdbid = clip_df.iloc[0]["imdbid_pe"]
            text_gt = clip_df.iloc[0]['text']

            # The AD interval
            gt_start = clip_df["AD_start"].iloc[0]
            gt_end = clip_df["AD_end"].iloc[0]

            # Video path
            seg_name = clip_df.iloc[0]["tvad_name"]
            if "friends" in seg_name:
                seg_path = os.path.join(video_dir, "friends_frames", seg_name + ".tar")
            else:
                seg_path = os.path.join(video_dir, "bbt_frames", seg_name + ".tar")
            if not os.path.exists(seg_path):
                continue
        
            # Predicted thread structures
            threads = ast.literal_eval(clip_df.iloc[0]["thread"])
            
            # Per-shot information
            ad_indices = clip_df["AD_index"].iloc[0]
            shot_indices = []
            shot_bboxes = []
            shot_pred_ids = []
            shot_labels = []
            shot_scales = []
            for shot_idx, shot_row in clip_df.iterrows():
                shot_indices.append(ast.literal_eval(shot_row["scaled_index"]))
                shot_bboxes.append(shot_row["bboxes"])
                shot_pred_ids.append(shot_row["pred_ids"])
                shot_labels.append(shot_row["shot_label"])
                shot_scales.append(shot_row["shot_scale"])
            self.all_clips.append([imdbid, anno_idx, seg_name, seg_path, threads, text_gt, gt_start, gt_end, ad_indices, shot_indices, shot_bboxes, shot_pred_ids, shot_labels, shot_scales])
        print(f"In total {len(self.all_clips)} TV-AD clips")

    def __len__(self):
        return len(self.all_clips)

    
    def __getitem__(self, index):
        imdbid, anno_idx, seg_name, video_path, threads, text_gt, gt_start, gt_end, ad_indices, shot_indices, shot_bboxes, shot_pred_ids, shot_labels, shot_scales = self.all_clips[index]
        
        '''
        Frame indices sampling: 16 frames current shots (including AD intervals) + 16 frames past & future shot frames
        This is slightly different from CMDAD and MADEval setup. As the TVAD frames are provided at 3 fps (no point to sample it denser than 3fps), we choose not to sample 16 frames in current shot frames rather than in AD intervals
        '''
        shot_boundary = [e[-1] + 0.5 for e in shot_indices[:-1]]
        ad_indices = ast.literal_eval(ad_indices)

        # Current shot sampling
        current_shots = [j for j, e in enumerate(shot_labels) if "m" in e]
        ad_indices = [e for e in ad_indices if (e >= shot_indices[current_shots[0]][0]) and (e <= shot_indices[current_shots[-1]][-1])]
        current_timearray = uniform_sampling(shot_indices[current_shots[0]][0], shot_indices[current_shots[-1]][-1], self.num_frames)
        current_split_timearray = group_values(current_timearray, shot_boundary)
        
        # Context shot sampling
        remaining_frames = self.num_frames
        past_shots = [j for j, e in enumerate(shot_labels) if "l" in e]
        future_shots = [j for j, e in enumerate(shot_labels) if "r" in e]
        if len(past_shots) == 0 and len(future_shots) == 0:
            split_timearray = current_split_timearray
        elif len(past_shots) == 0 and len(future_shots) != 0:
            future_timearray = uniform_sampling(shot_indices[future_shots[0]][0], shot_indices[future_shots[-1]][-1], remaining_frames)
            future_split_timearray = group_values(future_timearray, shot_boundary)
            split_timearray = [np.concatenate([j, k], 0) for j, k in zip(current_split_timearray, future_split_timearray)]
        elif len(past_shots) != 0 and len(future_shots) == 0:
            past_timearray = uniform_sampling(shot_indices[past_shots[0]][0], shot_indices[past_shots[-1]][-1], remaining_frames)
            past_split_timearray = group_values(past_timearray, shot_boundary)
            split_timearray = [np.concatenate([j, k], 0) for j, k in zip(past_split_timearray, current_split_timearray)]
        else:
            past_durations = shot_indices[past_shots[-1]][-1] - shot_indices[past_shots[0]][0] + 1
            future_durations = shot_indices[future_shots[-1]][-1] - shot_indices[future_shots[0]][0] + 1
            past_frames = int(remaining_frames * past_durations / (past_durations + future_durations + 1e-8))
            future_frames = remaining_frames - past_frames
            past_timearray = uniform_sampling(shot_indices[past_shots[0]][0], shot_indices[past_shots[-1]][-1], past_frames)
            past_split_timearray = group_values(past_timearray, shot_boundary)
            future_timearray = uniform_sampling(shot_indices[future_shots[0]][0], shot_indices[future_shots[-1]][-1], future_frames)
            future_split_timearray = group_values(future_timearray, shot_boundary)
            split_timearray = [np.concatenate([i, j, k], 0) for i, j, k in zip(past_split_timearray, current_split_timearray, future_split_timearray)]

        '''
        Read frames (with visual prompts)
        '''
        cache = None  # decord cache 
        all_frames = [] # all processed frames
        current_name_ids = {} # character information for current shots
        all_name_ids = {} # all character information
        shot_labels_refined = []
        shot_scales_refined = []
        shot_idx_tmp = 0
        valid_shots = {}
        for shot_idx, split_timearray_single in enumerate(split_timearray):
            if len(split_timearray_single) == 0:
                all_frames.extend([])
                valid_shots[shot_idx] = -1
                continue
            
            # Match with character-recognised frames
            shot_indices_single = shot_indices[shot_idx]
            selected_idx = [np.argmin(np.abs(np.array(shot_indices_single) - b)) for b in split_timearray_single]
            selected_timearray = split_timearray_single

            # Read frames
            bboxes = shot_bboxes[shot_idx]
            pred_ids = shot_pred_ids[shot_idx]
            frames, all_name_ids, current_name_ids, processed_frames = self.read_and_label_shot(shot_idx_tmp, ad_indices, selected_idx, selected_timearray, video_path, bboxes, pred_ids, all_name_ids, current_name_ids, shot_labels[shot_idx])
            all_frames.extend(processed_frames)

            # Since some shots are too short in time, which are not sampled at all, we need to correct the shot scales and thread structures accordingly
            shot_labels_refined.append(shot_labels[shot_idx])
            shot_scales_refined.append(shot_scales[shot_idx])
            valid_shots[shot_idx] = shot_idx_tmp
            shot_idx_tmp += 1

        # Refine all shot information (due to occational unsampled shots)
        current_shots_refined = [j for j, e in enumerate(shot_labels_refined) if "m" in e]
        threads = [e for e in threads if len(e) > 1]
        threads_refined = []
        for thread in threads:
            thread_refined = []
            for e in thread:
                if valid_shots[e] != -1:
                    thread_refined.append(valid_shots[e])
            if len(thread_refined) > 1:
                threads_refined.append(thread_refined)

        if self.model == "qwen2vl":
            # Convert to tensors
            all_frames = np.stack(all_frames, 0)
            video_tensor = torch.from_numpy(all_frames).permute(0, 3, 1, 2)
            video = fetch_video_from_tensor(video_tensor) 
        elif self.model == "gpt4o":  
            # Convert to base64 images    
            base64_images = []
            for frame_idx, processed_frame in enumerate(all_frames):
                base64_images.append(pil_to_base64_jpg(self.transform(processed_frame)))
            video = base64_images
        else:
            print("please specify 'qwen2vl' or 'gpt4o' as the model")
            sys.exit()
      
        '''
        Formulate character information and text prompts
        '''
        charbank_dict = self.charbank_dict[imdbid]
        all_char_info = []
        all_char_text = ". Possible characters (labeled by {label_type}): "
        for name_idx, (name_id, color_idx) in enumerate(all_name_ids.items()):
            if name_idx == len(all_name_ids) - 1:
                ending = ""
            else:
                ending = ", "
            all_char_text = all_char_text + charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")" + ending  
            all_char_info.append(charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")")
        
        # Split character info for current shots and context shots
        # This step is different from CMDAD and MADEval, as in TV series, there are much more characters involved
        # Therefore, we choose to emphasize the characters in current shots
        if all_char_text == ". Possible characters (labeled by {label_type}): " or self.label_type == "none": 
            char_text = ""
        else:
            current_char_info = []
            for name_idx, (name_id, color_idx) in enumerate(current_name_ids.items()):
                if name_idx == len(current_name_ids) - 1:
                    ending = ""
                else:
                    ending = ", "
                current_char_info.append(charbank_dict[name_id] + " (" + self.color_name[color_idx] + ")")
            context_char_info = [element for element in all_char_info if element not in current_char_info]
            
            current_char_info_text = ", ".join(current_char_info)
            context_char_info_text = ", ".join(context_char_info)

            if current_char_info == [] and context_char_info == []:
                char_text = ""
            elif current_char_info == [] and context_char_info != []:
                char_text = f". Possible characters (labeled by {self.label_type}): {context_char_info_text}"
            elif current_char_info != [] and context_char_info == []:
                char_text = f". Possible characters in targeted shots (labeled by {self.label_type}): {current_char_info_text}"
            else:
                char_text = f". Possible characters in targeted shots (labeled by {self.label_type}): {current_char_info_text}. Other possible characters (labeled by {self.label_type}): {context_char_info_text}"

        if self.model == "qwen2vl":
            processed_text_prompt, text_prompt = process_text_prompt(self.general_prompt.apply(char_text, current_shots=current_shots_refined, threads=threads_refined, shot_scales=shot_scales_refined), self.processor)
        elif self.model == "gpt4o": 
            text_prompt = self.general_prompt.apply(char_text, current_shots=current_shots_refined, threads=threads_refined, shot_scales=shot_scales_refined)
            processed_text_prompt = text_prompt
        else:
            print("please specify 'qwen2vl' or 'gpt4o' as the model")
            sys.exit()

        return_dict =  {'video': video, \
                'imdbid': imdbid, \
                'processed_prompt': processed_text_prompt,
                'prompt': text_prompt,
                'gt_text': text_gt,
                'start': gt_start,  # AD interval start
                'end': gt_end,  # AD interval end
                'start_': gt_start, # AD interval start (repeated to match format)
                'end_': gt_start, # AD interval end (repeated to match format)
                'anno_idx': anno_idx,
                }
        return return_dict
    
    @staticmethod
    def collate_fn(batch):
        out_batch = {}
        out_batch['imdbid'] = [sample['imdbid'] for sample in batch]
        out_batch['video'] = [sample['video'] for sample in batch]
        out_batch['start'] = [sample['start'] for sample in batch]
        out_batch['end'] = [sample['end'] for sample in batch]
        out_batch['start_'] = [sample['start_'] for sample in batch]
        out_batch['end_'] = [sample['end_'] for sample in batch]
        out_batch['processed_prompt'] = [sample['processed_prompt'] for sample in batch]
        out_batch['prompt'] = [sample['prompt'] for sample in batch]
        out_batch['gt_text'] = [sample['gt_text'] for sample in batch]
        out_batch['anno_idx'] = [sample['anno_idx'] for sample in batch]
        return out_batch


    def read_and_label_shot(self, shot_idx, real_indices, selected_idx, selected_timearray, video_path, bboxes, name_ids, all_name_ids, current_name_ids, shot_label):
        '''
        Read selected frames for each shot
        Add colour-coded circles/boxes around character faces
        Add shot labels
        (Optional) Add outline box for AD interval frames
        '''
        # Read frames and add character face labels
        frames = []
        for img_idx in selected_timearray:
            frames.append(read_tarfile(video_path, int(img_idx)))
        if self.label_type == "none":
            processed_frames = frames
        else:
            bboxes = ast.literal_eval(bboxes)
            name_ids = ast.literal_eval(name_ids)  # {"ttxxxxxx": 1, ...]
            named_bboxes = [{} for i in range(len(bboxes))]  # [{"ttxxxxxx": [bbox], ...}, {}, {}, ...]
            for name_id, bbox_info in name_ids.items():
                for bbox_info_single in bbox_info:
                    named_bboxes[bbox_info_single[0]][name_id] = bboxes[bbox_info_single[0]][bbox_info_single[1]]

            bboxes_filtered = np.array(named_bboxes, dtype=object)[selected_idx].tolist()
            shot_name_ids = extract_ordered_keys(bboxes_filtered)
            for shot_name_id in shot_name_ids:
                if shot_name_id not in all_name_ids.keys():
                    all_name_ids[shot_name_id] = len(all_name_ids)

            if "m" in shot_label: # if it is a "current shot"
                for shot_name_id in shot_name_ids:
                    if shot_name_id not in current_name_ids.keys():
                        current_name_ids[shot_name_id] = all_name_ids[shot_name_id]

            processed_frames = []
            for frame_idx, frame in enumerate(frames):
                if len(bboxes_filtered[frame_idx]) == 0: 
                    processed_frames.append(frame)
                else:
                    label_masks = None
                    total_masks = None
                    for b_idx, (name_id, bbox) in enumerate(bboxes_filtered[frame_idx].items()):
                        # Draw binary label masks
                        if self.label_type=="boxes":
                            label_mask = convert_bounding_box_to_rectangle(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        elif self.label_type=="circles":
                            label_mask = convert_bounding_box_to_ellipse(bbox, canvas_width=frame.size[0], canvas_height=frame.size[1], line_width=int(self.label_width/1000*frame.size[0]))
                        else:
                            print("Check the label type")
                            sys.exit()
                        # Overlay label masks to get an overall mask
                        if label_masks is None:
                            label_masks = label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = label_mask
                        else:
                            label_masks = label_masks * (1 - label_mask[:, :, None]) + label_mask[:, :, None] * np.array(self.colors[all_name_ids[name_id]])[None, None, :]
                            total_masks = np.clip(label_mask + total_masks, 0., 1.)
                    # Overlay the overall label mask onto the frame
                    processed_frame = Image.fromarray((np.array(frame) * (1- total_masks[:, :, None] * self.label_alpha) + total_masks[:, :, None] * self.label_alpha * label_masks).astype(np.uint8))            
                    processed_frames.append(processed_frame)
        
        # Add shot numbers
        if self.shot_label:
            final_frames = []
            for frame_idx, processed_frame in enumerate(processed_frames):
                final_frames.append(self.draw_shot_label(shot_idx, processed_frame))
        else:
            final_frames = processed_frames

        # Add outline boxes
        if self.adframe_label:
            final_frames_ = []
            for frame_idx, processed_frame in enumerate(final_frames):
                selected_time = selected_timearray[frame_idx]
                if int(selected_time) in real_indices:
                    final_frames_.append(self.add_red_border(processed_frame, border_thickness=7))
                else:
                    final_frames_.append(processed_frame)
        else:
            final_frames_ = final_frames

        return frames, all_name_ids, current_name_ids, final_frames_


    def draw_shot_label(self, shot_idx, image):
        # Create a drawing context
        draw = ImageDraw.Draw(image)

        # Set font (default to a simple font if PIL cannot find a specific one)
        font = ImageFont.truetype(self.font_path, 40)

        # Set text
        text = f"Shot {shot_idx}"

        # Calculate text size to draw a background rectangle
        text_bbox = draw.textbbox((0, 0), text, font=font)  # Get bounding box of the text
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        padding = 7
        background_position = (0, 0, text_width + 2 * padding, text_height + 4 * padding)

        # Draw white rectangle as background for the text
        draw.rectangle(background_position, fill="white")

        # Draw the text in black on top of the white rectangle
        text_position = (padding, padding)
        draw.text(text_position, text, fill="black", font=font)

        return image


    def add_red_border(self, image, border_thickness=5):
        # Create a drawing context
        draw = ImageDraw.Draw(image)
        
        # Get image dimensions
        width, height = image.size
        
        # Draw a red rectangle as the border
        for i in range(border_thickness):
            draw.rectangle(
                [i, i, width - i - 1, height - i - 1], outline="red"
            )
        
        return image

