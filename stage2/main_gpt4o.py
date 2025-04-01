import os
import sys
import ast
import json
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from promptloader import get_user_prompt

from openai import OpenAI
os.environ["OPENAI_API_KEY"] = #TODO

def summary_each(client, user_prompt, dataset, idx, mode):      
    if dataset in ["cmdad", "madeval"]:
        dataset_text = "movie"
    elif dataset in ["tvad"]:
        dataset_text = "TV series"

    sys_prompt = (
            f"You are an intelligent chatbot designed for summarizing {dataset_text} audio descriptions. "
            "Here's how you can accomplish the task:------##INSTRUCTIONS: you should convert the predicted descriptions into one sentence. "
            "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end."
    )

    messages = [
        {
            "role": "system",
            "content": sys_prompt  
        },
        {
            "role": "user",
            "content": user_prompt  
        }
    ]
    
    repeat = True
    iters = 0 
    while repeat and iters < 5:
        if iters >=1:
            print("Redo due to error")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        output_text = response.choices[0].message.content
        if mode == "single" and "{\"summarized_AD\": \"" in output_text:
            repeat = False
        else: # assistant mode
            for ad_idx in range(1, 6):
                if f"\"summarized_AD_{ad_idx}\":" not in output_text:
                    repeat = True
                    continue
                repeat = False
    return output_text


def main(args):
    # Initialise the openai client
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Read predicted output from Stage I
    pred_df = pd.read_csv(args.pred_path)

    # Dataset-specific information
    if args.dataset in ["cmdad"]:
        gt_df = pd.read_csv("gt_ad_train/cmdad_train.csv") # GT ADs in training split
        verb_list = ['look', 'turn', 'take', 'hold', 'pull', 'walk', 'run', 'watch', 'stare', 'grab', 'fall', 'get', 'go', 'open', 'smile']
        ad_speed = 0.275
       
    elif args.dataset in ["tvad"]:
        gt_df = pd.read_csv("gt_ad_train/tvad_train.csv") # GT ADs in training split
        verb_list = ['look', 'walk', 'turn', 'stare', 'take', 'hold', 'smile', 'leave', 'pull', 'watch', 'open', 'go', 'step', 'get', 'enter']
        ad_speed = 0.2695

    elif args.dataset in ["madeval"]:
        gt_df = pd.read_csv("gt_ad_train/madeval_train.csv") # GT ADs in training split
        verb_list = ['look', 'turn', 'sit', 'walk', 'take', 'stand', 'watch', 'hold', 'pull', 'see', 'go', 'open', 'smile', 'run', 'get']
        ad_speed = 0.5102 
    else:
        print("Check the dataset name")
        sys.exit()

    # Extract GT AD list (w & wo character information)
    all_gts = gt_df["text_gt"].tolist()
    all_gts_wo_char = gt_df["text_gt_wo_char"].tolist()
    all_gts_num_words = [len(e.strip().split(" ")) for e in all_gts_wo_char]

    text_gen_list = []
    text_gt_list = []
    start_sec_list = []
    end_sec_list = []
    imdbid_list = []
    anno_indices = []
    for row_idx, row in tqdm(pred_df.iterrows(), total=len(pred_df)):
        # Estimate the number of words based on training split statistics
        duration = round(row['end'] - row['start'], 2)
        rough_num_words = round(duration / ad_speed)

        text_gt = row['text_gt']
        text_pred = str(row['text_gen'])

        # Sample GT ADs with roughly the same length as examples (+-1 word)
        candid_indices = [i for i, s in enumerate(all_gts_num_words) if rough_num_words - 1 <= s <= rough_num_words + 1]
        if len(candid_indices) < args.num_examples:
            candid_indices = list(range(len(all_gts_num_words)))
        sampled_indices = random.choices(candid_indices, k=args.num_examples)
        sampled_examples = [all_gts_wo_char[index] for index in sampled_indices]

        # Formulate the user prompt
        if args.dataset == "madeval": # As gpt4o follows the word limit too strictly, and the duration and num_words in MADEval is not exact linearly correlated, relax the word limit
            user_prompt = get_user_prompt(mode=args.mode, prompt_idx=args.prompt_idx, verb_list=verb_list, text_pred=text_pred, word_limit=int(duration/ad_speed)+3, examples=sampled_examples)
        else:
            user_prompt = get_user_prompt(mode=args.mode, prompt_idx=args.prompt_idx, verb_list=verb_list, text_pred=text_pred, word_limit=int(duration/ad_speed)+1, examples=sampled_examples)
        
        # Output AD
        text_summary = summary_each(client, user_prompt, args.dataset, row_idx, args.mode)
        try:
            if args.mode == "single": # default single AD mode
                text_summary = text_summary.replace("{\"summarized_AD\": \"", "").replace("\"}", "").strip()
                if "." != text_summary[-1]:  # Add comma if not existing
                    text_summary = text_summary + "."
                output_ads = text_summary
            else: # assistant mode (predict 5 AD candidates)
                output_ad_list = []
                for ad_idx in range(1, 6):
                    if "\n" not in text_summary:
                        text_summary_tmp = text_summary.split(f"\"summarized_AD_{ad_idx}\":")[-1].split(", \"")[0].split(",\"")[0].replace('\"', "").replace('{', "").replace('}', "").strip()
                    else:
                        text_summary_tmp = text_summary.split(f"\"summarized_AD_{ad_idx}\":")[-1].split(",\n")[0].split("\n")[0].replace('\"', "").replace('{', "").replace('}', "").strip()
                    if "." != text_summary_tmp[-1]: 
                        text_summary_tmp = text_summary_tmp + "."
                    output_ad_list.append(text_summary_tmp)
                output_ads = str(output_ad_list)
        except: 
            output_ads = ""

        print(output_ads)
        text_gen_list.append(output_ads)
        text_gt_list.append(text_gt)
        start_sec_list.append(row['start'])
        end_sec_list.append(row['end'])
        imdbid_list.append(row['imdbid'])
        anno_indices.append(row['anno_idx'])


    output_df = pd.DataFrame.from_records({'imdbid': imdbid_list, 'start': start_sec_list, 'end': end_sec_list, 'text_gt': text_gt_list, 'text_gen': text_gen_list, 'anno_idx': anno_indices})
    save_path = os.path.join(args.save_dir, args.dataset + "_ads", f"stage2_gpt4o_{args.mode}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_df.to_csv(save_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default=None, type=str, help='input directory')
    parser.add_argument('--save_dir', default=None, type=str, help='output directory')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--mode', default="single", type=str)
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--prompt_idx', default=0, type=int, help='optional, use to indicate you own prompt')
    parser.add_argument('--num_examples', default=10, type=int, help='number of GT ADs')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
   