import os
os.environ['TRANSFORMERS_CACHE'] = #TODO
import sys
import ast
import json
import torch
import random
import argparse
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
from promptloader import get_user_prompt


def initialise_model(access_token):
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
        token=access_token,
    )
    return pipeline

def summary_each(pipeline, user_prompt, dataset):      
    if dataset in ["cmdad", "madeval"]:
        dataset_text = "movie"
    elif dataset in ["tvad"]:
        dataset_text = "TV series"

    sys_prompt = (
            f"[INST] <<SYS>>\nYou are an intelligent chatbot designed for summarizing {dataset_text} audio descriptions. "
            "Here's how you can accomplish the task:------##INSTRUCTIONS: you should convert the predicted descriptions into one sentence. "
            "You should directly start the answer with the converted results WITHOUT providing ANY more sentences at the beginning or at the end. \n<</SYS>>\n\n{} [/INST] "
    )

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = pipeline.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id = pipeline.tokenizer.eos_token_id,
    )

    output_text = outputs[0]["generated_text"][len(prompt):]
    return output_text

def main(args):
    # Initialise the model
    pipeline = initialise_model(args.access_token)
   
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
    all_gts_num_words = [len(str(e).strip().split(" ")) for e in all_gts_wo_char]

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
        user_prompt = get_user_prompt(mode=args.mode, prompt_idx=args.prompt_idx, verb_list=verb_list, text_pred=text_pred, word_limit=int(duration/ad_speed)+1, examples=sampled_examples)
        
        # Output AD
        text_summary = summary_each(pipeline, user_prompt, args.dataset)
        try:
            if args.mode == "single": # default single AD mode
                text_summary = text_summary.replace("{\"summarized_AD\": \"", "").replace("\"}", "").strip()
                if "." != text_summary[-1]:  # Add comma if not existing
                    text_summary = text_summary + "."
                output_ads = text_summary
            else: # assistant mode (predict 5 AD candidates)
                output_ad_list = []
                for ad_idx in range(1, 6):
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
        # import ipdb; ipdb.set_trace()


    output_df = pd.DataFrame.from_records({'imdbid': imdbid_list, 'start': start_sec_list, 'end': end_sec_list, 'text_gt': text_gt_list, 'text_gen': text_gen_list, 'anno_idx': anno_indices})
    save_path = os.path.join(args.save_dir, args.dataset + "_ads", f"stage2_llama3_{args.mode}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    output_df.to_csv(save_path, index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path', default=None, type=str, help='input directory')
    parser.add_argument('--save_dir', default=None, type=str, help='output directory')
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--mode', default="single", type=str)
    parser.add_argument('--access_token', default=None, type=str, help='HuggingFace token to access llama3')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--prompt_idx', default=0, type=int, help='optional, use to indicate you own prompt')
    parser.add_argument('--num_examples', default=10, type=int, help='number of GT ADs')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    main(args)
   