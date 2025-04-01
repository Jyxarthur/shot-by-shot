import os
os.environ['TRANSFORMERS_CACHE'] = #TODO
import ast
import sys
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from promptloader import PromptLoader
from dataloader import CMDAD_Dataset, TVAD_Dataset, MADEval_Dataset
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor


def main(args):
    # Load model
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    
    # Build dataloader
    if args.dataset == "cmdad":
        D = CMDAD_Dataset 
        video_type = "movie"
    elif args.dataset == "tvad":
        D = TVAD_Dataset
        video_type = "TV series"
    elif args.dataset == "madeval":
        D = MADEval_Dataset
        video_type = "movie"
        args.num_workers = 8
    else:
        print("Check dataset name")
        sys.exit()

    # Formulate text prompt template
    general_prompt = PromptLoader(prompt_idx=args.prompt_idx, video_type=video_type, label_type=args.label_type)

    ad_dataset = D(model="qwen2vl", processor=processor, general_prompt=general_prompt, num_frames=args.num_frames, 
                    anno_path=args.anno_path, charbank_path=args.charbank_path, video_dir=args.video_dir, font_path=args.font_path,
                    label_type=args.label_type, label_width=args.label_width, label_alpha=args.label_alpha, 
                    prompt_idx=args.prompt_idx, adframe_label=args.adframe_label, shot_label=args.shot_label)
    
    loader = torch.utils.data.DataLoader(ad_dataset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                            collate_fn=ad_dataset.collate_fn, shuffle=False, pin_memory=True)

    start_sec = []
    end_sec = []
    start_sec_ = []
    end_sec_ = []
    text_gt = []
    text_gen = []
    imdbids = []
    anno_indices = []

    for idx, input_data in tqdm(enumerate(loader), total=len(loader), desc='EVAL'): 
        video_inputs = input_data["video"]
        texts = input_data["processed_prompt"]

        # Inference
        inputs = processor(
            text=texts,
            images=None,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")     
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # Remove unneeded texts
        outputs = [output.split("\n\nExplanation:")[0] for output in outputs] 
        outputs = [output.split("\nExplanation:")[0] for output in outputs] 
        outputs = [output.split("Explanation:")[0] for output in outputs] 
        outputs = [output.split("Description:\n\n")[-1] for output in outputs] 
        outputs = [output.split("Description:\n")[-1] for output in outputs] 
        outputs = [output.split("Description:")[-1] for output in outputs] 
        print(outputs)
            
        anno_indices.extend(input_data["anno_idx"])
        imdbids.extend(input_data["imdbid"])
        text_gt.extend(input_data["gt_text"])
        text_gen.extend(outputs) 
        start_sec.extend(input_data["start"])
        end_sec.extend(input_data["end"])
        start_sec_.extend(input_data["start_"])
        end_sec_.extend(input_data["end_"])
       
    # Saving
    output_df = pd.DataFrame.from_records({'anno_idx': anno_indices, 'imdbid': imdbids, 'start': start_sec, 'end': end_sec, 'start_': start_sec_, 'end_': end_sec_, 'text_gt': text_gt, 'text_gen': text_gen})
    save_path = os.path.join(args.save_dir, f"{args.dataset}_ads", "stage1_qwen2vl.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_df.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Base
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--num_frames', default=16, type=int, help='number of frames')
    parser.add_argument('-j', '--num_workers', default=8, type=int, help='init mode')
    parser.add_argument('--seed', default=42, type=int, help='evaluation seed')
    # Inputs
    parser.add_argument('--anno_path', default=None, type=str)
    parser.add_argument('--charbank_path', default=None, type=str)
    parser.add_argument('--video_dir', default=None, type=str)
    parser.add_argument('--save_dir', default=None, type=str)
    parser.add_argument('--font_path', default=None, type=str)
    # Label setup
    parser.add_argument('--label_type', default="circles", type=str)
    parser.add_argument('--label_width', default=10, type=int, help='label_width, 10 in a canvas 1000')
    parser.add_argument('--label_alpha', default=0.8, type=float)
    parser.add_argument('--shot_label', action='store_true', help='shot number on the top left')
    parser.add_argument('--adframe_label', action='store_true', help='AD interval frames outlined by red boxes')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    main(args)

    
    