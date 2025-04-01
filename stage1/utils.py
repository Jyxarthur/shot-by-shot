import os
import cv2
import sys
import copy
import torch
import base64
import tarfile
import numpy as np

from PIL import Image
from io import BytesIO
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from qwen_vl_utils import smart_resize

# Base setup from qwen2-vl
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


def fetch_video_from_tensor(video, image_factor: int = IMAGE_FACTOR):
    nframes, _, height, width = video.shape
    ele = {}
    ele["max_pixels"] = 360 * 420
    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = ele.get("max_pixels", max_pixels)
    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"],
            ele["resized_width"],
            factor=image_factor,
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    video = transforms.functional.resize(
        video,
        [resized_height, resized_width],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video


def process_video(video_path, num_frames=8, sample_scheme='uniform', start = None, end = None, cache = None):
    """
    Input: video path, start & end time
    Output: a list of 8 consecutive PIL images
    """
    decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, num_frames, endpoint=False, dtype=int)
    else:
        frame_id_list = frame_sample(duration, num_frames, mode=sample_scheme, local_fps=local_fps)
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
    return images, decord_vr


def process_video_selected_idx(video_path, selected_idx, sample_scheme='uniform', start = None, end = None, cache = None):
    """
    Input: video path, start & end time
    Output: a list of 8 consecutive PIL images
    """
    decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    # import ipdb; ipdb.set_trace()
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, 32, endpoint=False, dtype=int)
    else:
        frame_id_list = frame_sample(duration, 32, mode=sample_scheme, local_fps=local_fps)
    frame_id_list = frame_id_list[selected_idx]
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
    return images, decord_vr


def process_video_selected_idx_cache(video_path, selected_idx, sample_scheme='uniform', start = None, end = None, cache = None):
    """
    Input: video path, start & end time
    Output: a list of 8 consecutive PIL images
    """
    if cache is not None:
        decord_vr = cache
    else:
        decord_vr = VideoReader(uri=video_path, ctx=cpu(0)) 
    duration, local_fps = len(decord_vr), float(decord_vr.get_avg_fps())
    # import ipdb; ipdb.set_trace()
    if start is not None and end is not None:
        start_frame, end_frame = local_fps * start, local_fps * end
        end_frame = min(end_frame, len(decord_vr) - 1)
        frame_id_list = np.linspace(start_frame, end_frame, 32, endpoint=False, dtype=int)
    else:
        frame_id_list = frame_sample(duration, 32, mode=sample_scheme, local_fps=local_fps)
    frame_id_list = frame_id_list[selected_idx]
    try:
        video_data = decord_vr.get_batch(frame_id_list).numpy()
    except:
        video_data = decord_vr.get_batch(frame_id_list).asnumpy()
    images = [Image.fromarray(f.numpy() if isinstance(f, torch.Tensor) else f) for f in video_data]
    return images, decord_vr


def frame_sample(duration, num_frames, mode='uniform', local_fps=None):
    if mode == 'uniform':
        return np.linspace(0, duration-1, num_frames, dtype=int)
    elif mode == 'fps':
        assert local_fps is not None
        segment_len = min(local_fps // NUM_FRAMES_PER_SECOND, duration)
        return np.arange(segment_len // 2, duration, segment_len, dtype=int)
    else:
        raise ImportError(f'Unsupported frame sampling mode: {mode}')


def convert_bounding_box_to_rectangle(bbox, canvas_width=1280, canvas_height=720, line_width=10):
    """
    Input: bbox, canvas (image) size
    Output: binary bbox mask
    """
    x0, y0, x1, y1 = bbox

    # extend the bbox by 20%
    delta_x = (x1 - x0) * 0.1
    delta_y = (y1 - y0) * 0.1

    x0 = int(max(x0 - delta_x, 0))
    y0 = int(max(y0 - delta_y, 0))
    x1 = int(min(x1 + delta_x, canvas_width))
    y1 = int(min(y1 + delta_y, canvas_height))
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas[y0:y0+line_width, x0:x1] = 1. 
    canvas[y1-line_width:y1, x0:x1] = 1. 
    canvas[y0:y1, x0:x0+line_width] = 1.  
    canvas[y0:y1, x1-line_width:x1] = 1. 
    return canvas


def convert_bounding_box_to_ellipse(bbox, canvas_width=1280, canvas_height=720, line_width=10):
    """
    Input: bbox, canvas (image) size
    Output: binary ellipse mask
    """
    x0, y0, x1, y1 = bbox
    x0 = int(x0)
    y0 = int(y0)
    x1 = int(x1)
    y1 = int(y1)
    center_x = (x0 + x1) // 2
    center_y = (y0 + y1) // 2

    # extend the ellipse by 25%
    axis_x = int(abs(x1 - x0) // 2 * 1.25)
    axis_y = int(abs(y1 - y0) // 2 * 1.25)
    
    canvas = np.zeros((canvas_height, canvas_width), dtype=np.float32)
    canvas = cv2.ellipse(canvas, (center_x, center_y), (axis_x, axis_y), 0, 0, 360, 1, line_width)
    return canvas

def process_text_prompt(question, processor):
    msg = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": ""},
                {"type": "text", "text": question},
            ],
        }
    ]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    return text, question


def read_tarfile(tar_path, img_idx):
    """
    Input: tarfile path, image index to extract
    Output: an PIL image
    """
    dirname = os.path.basename(tar_path.replace(".tar", ""))
    imagename = os.path.join(dirname, str(img_idx).zfill(5) + ".jpg")
    with tarfile.open(tar_path, 'r') as tar:
        fileobj = tar.extractfile(imagename)
        image_data = fileobj.read()
        image_tmp = Image.open(BytesIO(image_data))
        image = copy.deepcopy(image_tmp)
        image_tmp.close()
        fileobj.close()
    return image


def partition_array(array, separators):
    if len(separators) == 0:
        return [array]
    separators = sorted(separators)
    partitions = []
    partitions.append(array[array < separators[0]])
    for i in range(1, len(separators)):
        partitions.append(array[(array >= separators[i - 1]) & (array < separators[i])])
    partitions.append(array[array >= separators[-1]])
    return partitions


def group_values(values, boundaries):
    result = [[] for _ in range(len(boundaries) + 1)]

    if len(boundaries) == 0:
        return [values]

    for value in values:
        placed = False
        for i, boundary in enumerate(boundaries):
            if value <= boundary:
                result[i].append(value)
                placed = True
                break
        if not placed:
            result[-1].append(value)
    return result


def get_context_timearray(num_frames, past_shots, future_shots, shot_starts, shot_ends, shot_boundary):
    '''
    Find the timestamps for past and future shots
    '''
    if len(past_shots) != 0:
        past_start = shot_starts[past_shots[0]]
        past_end = shot_ends[past_shots[-1]]
        past_duration = past_end - past_start
    else:
        past_start = 0
        past_end = 0
        past_duration = 0

    if len(future_shots) != 0:
        future_start = shot_starts[future_shots[0]]
        future_end = shot_ends[future_shots[-1]]
        future_duration = future_end - future_start
    else:
        future_start = 0
        future_end = 0
        future_duration = 0

    # Calculate the number of frames for past and future shots
    if (past_duration + future_duration) != 0:
        past_num_frames = int(num_frames * past_duration / (past_duration + future_duration))
        future_num_frames = num_frames - past_num_frames
    else:
        past_num_frames = 0
        future_num_frames = 0

    if past_num_frames != 0:
        past_timearray = np.linspace(past_start, past_end, past_num_frames, endpoint=False)
        past_split_timearray = partition_array(past_timearray, shot_boundary)
    else:
        past_split_timearray = []
    
    if future_num_frames != 0:
        future_timearray = np.linspace(future_start, future_end, future_num_frames, endpoint=False)
        future_split_timearray = partition_array(future_timearray, shot_boundary)
    else:
        future_split_timearray = []
    return past_split_timearray, future_split_timearray


def uniform_sampling(start, end, num_frames):
    result = np.clip(np.round(np.linspace(start - 0.499, end + 0.499, num_frames)), start, end)
    return result.tolist()


def extract_ordered_keys(data):
    ordered_keys = []
    for dictionary in data:
        for key in dictionary:
            if key not in ordered_keys:
                ordered_keys.append(key)
    return ordered_keys


def pil_to_base64_jpg(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")  # Set format to JPEG
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def process_add_e(add_e):
    if len(add_e) > 0 and add_e[-1] not in [".", ";"]:
        if add_e[-1] == " ":
            add_e = add_e[:-1]
        if len(add_e) == 0:
            return None
        add_e = add_e + "."
    return add_e

def output_cleaning(clean_result):
    '''
    If the descriptions are provided for different shots independently (This exclusively happens to GPT4o), post-process to group them together
    '''
    if "1. Main characters: " in clean_result:
        if len(clean_result.split("1. Main characters: ")) == 2:
            clean_result = "1. Main characters: " + clean_result.split("1. Main characters: ")[-1].replace(" \n", "\n").replace(" \n", "\n").replace(" \n", "\n").replace(".\n\n", ". ").replace(";\n\n", "; ").replace(".\n", ". ").replace(";\n", "; ").replace("\n\n", ". ").replace("\n", ". ")
        else:
            factor_1 = "1. Main characters: "
            factor_2 = "2. Actions: "
            factor_3 = "3. Character-character interactions: "
            if "4. Key objects: " in clean_result:
                factor_4 = "4. Key objects: "
            elif "4. Facial expressions: " in clean_result:
                factor_4 = "4. Facial expressions: "
            elif "4. Environment: " in clean_result:
                factor_4 = "4. Environment: "
            else:
                factor_4 = None
            
            results_1 = []
            for e in clean_result.split(factor_1):
                if factor_2 in e:
                    add_e = e.split(factor_2)[0].replace("\n", "")
                    add_e = process_add_e(add_e)
                    if add_e is not None:
                        results_1.append(add_e)

            results_2 = []
            for e in clean_result.split(factor_2):
                if factor_3 in e:
                    add_e = e.split(factor_3)[0].replace("\n", "")
                    add_e = process_add_e(add_e)
                    if add_e is not None:
                        results_2.append(add_e)
            
            results_3 = []
            for i_, e in enumerate(clean_result.split(factor_3)):
                if factor_4 is not None and factor_4 in e:
                    add_e = e.split(factor_4)[0].replace("\n", "")
                    add_e = process_add_e(add_e)
                    if add_e is not None:
                        results_3.append(add_e)
                
                if factor_4 is None:
                    if i_ == 0:
                        continue
                    add_e = e.split(factor_1)[0].split("Shot ")[0].split("###")[0].replace("\n", "")
                    add_e = process_add_e(add_e)
                    if add_e is not None:
                        results_3.append(add_e)

            results_4 = []
            if factor_4 is not None:
                for i_, e in enumerate(clean_result.split(factor_4)):
                    if i_ == 0:
                        continue
                    add_e = e.split(factor_1)[0].split("Shot ")[0].split("###")[0].replace("\n", "")
                    add_e = process_add_e(add_e)
                    if add_e is not None:
                        results_4.append(add_e)
            if factor_4 is not None:
                clean_result = factor_1 + " ".join(results_1) + " " + factor_2 + " ".join(results_2) + " " + factor_3 + " ".join(results_3) + " " + factor_4 + " ".join(results_4)
            else:
                clean_result = factor_1 + " ".join(results_1) + " " + factor_2 + " ".join(results_2) + " " + factor_3 + " ".join(results_3)
        return clean_result.replace("**.", "").replace("**", "")
    else:
        return None
