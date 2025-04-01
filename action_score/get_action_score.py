import os
import re
import ast
import sys
import copy
import json
import torch
import argparse
import numpy as np
import transformers
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import spacy
nlp = spacy.load("en_core_web_lg")



# Spliting according to ";" and "."
def paragraph_spliter(paragraph): 
    sentences = []
    # Split the sentence by "." and ";"
    parts = re.split(r'[.;]', paragraph)

    # Print the resulting parts
    for part in parts:
        part = part.strip()
        if part in ["", "."]:
            continue
        sentences.append(part + ".")
    return sentences

# Spliting according to ";", ".", and ","
def paragraph_spliter_fine(paragraph):
    sentences = []
    # Split the paragraph by "." and ";"
    parts = re.split(r'[.;,]', paragraph)

    # Print the resulting parts
    for part in parts:
        part = part.strip()
        if part in ["", ".", ","]:
            continue
        sentences.append(part + ".")
    return sentences


def partition_list(input_list, chunk_size):
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def extract_action_phrases(sentence):
    doc = nlp(sentence)
    actions = []
    for token in doc:
        if token.pos_ == "VERB":  # Identify verbs
            # Extract subtree tokens for the verb
            phrase_tokens = []
            last_dep = ""
            # import ipdb; ipdb.set_trace()
            for child in token.subtree:
                # Skip tokens related to coordinated verbs
                if child.dep_ != "conj" or child.head != token:
                    phrase_tokens.append(child.text)
                    last_dep = child.dep_
                else:
                    break  # Stop at the conjunction
            if last_dep == "cc" or last_dep == "punct":
                phrase_tokens.pop()
            action = {
                "verb": token.text,
                "phrase": " ".join(phrase_tokens)
            }
            actions.append(action)
    return actions


def extract_verb_lemma(sentence):
    doc = nlp(sentence)
    for token in doc:
        if token.pos_ == "VERB":
            # Base verb (lemma)
            base_verb = token.lemma_
            # Check for particles (e.g., "up", "down") and combine
            particles = [child.text for child in token.children if child.dep_ == "prt"]
            output = " ".join([base_verb] + particles)
            if output == "be":
                continue
            else:
                return output
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=None, type=str)
    parser.add_argument('--chunk_size', default=40, type=int, help='Chunk size for GTE extraction, reduce if OOM')
    parser.add_argument('--mode', default="ad", type=str, help='paragraph or ad')
    parser.add_argument('--pred_path', default=None, type=str, help='input prediction')
    parser.add_argument('--save_path', default=None, type=str)
    args = parser.parse_args()

    # Load GTE model
    model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True).cuda()
    model.max_seq_length = 8192

    # Load prediction and GT actions
    df = pd.read_csv(args.pred_path)
    if args.dataset == "cmdad":
        ref_df = pd.read_csv("gt_actions/cmdad_gt_action.csv")
    elif args.dataset == "tvad":
        ref_df = pd.read_csv("gt_actions/tvad_gt_action.csv")
    elif args.dataset == "madeval":
        ref_df = pd.read_csv("gt_actions/madeval_gt_action.csv")
    else:
        print("please specify the dataset")
        sys.exit()
    df["gt_action"] = ref_df["gt_action"]
    result_df = df.copy()
    result_df["action_score_breakdown"] = -1
    result_df["action_score"] = -1
   

    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        # If there is errors in loading, skip the example
        if row["gt_action"].isdigit(): # error in gt action labels
            continue
        try:
            gt_actions = ast.literal_eval(row["gt_action"])
        except:
            continue
        if pd.isna(row["text_gen"]):
            continue
        if not isinstance(row["text_gen"], str):
            continue

        '''
        Hierarchical parsing
        '''
        # Split paragraph/sentences into subsentences
        if args.mode == "paragraph": # paragraph mode
            paragraph = row["text_gen"]
            # Customised rules to remove formatting structures
            paragraph = paragraph.replace("1. Main characters: ", "").replace("1. Main characters:", "").replace("Main characters:", "").replace("1.", "")
            paragraph = paragraph.split("2. Actions:")[-1].replace("Actions:", "").replace("2.", "")
            paragraph = paragraph.replace("3. Character-character interactions: ", "").replace("3. Character-character interactions", "").replace("Character-character interactions:", "").replace("3.", "")
            paragraph = paragraph.replace("4. Facial expressions: ", "").replace("4. Facial expressions:", "").replace("Facial expressions:", "").replace("4.", "")
            paragraph = paragraph.replace("###ANSWER TEMPLATE###: ", "").replace("###: ", "").replace("ANSWER TEMPLATE: ", "")
            pred_sentences = paragraph_spliter(paragraph)
            pred_subsentences = paragraph_spliter_fine(paragraph)
            pred_sentence_subsentences = list(set(pred_sentences + pred_subsentences))
        else: # ad mode
            pred_sentences = [row["text_gen"]]
            pred_subsentences = paragraph_spliter_fine(row["text_gen"])
            pred_sentence_subsentences = list(set(pred_sentences + pred_subsentences))

        # Extract predicted action phrases and verb lemma
        pred_actions = []
        for pred_subsentence in pred_subsentences:
            for e_ in extract_action_phrases(pred_subsentence):
                pred_actions.append(e_['phrase'] + ".")
        pred_verbs = [extract_verb_lemma(e) for e in pred_actions]


        # Combine all predictions
        pred_sentence_subsentences = [e for e in pred_sentence_subsentences if e not in pred_actions]
        all_preds = pred_actions + pred_sentence_subsentences
        if len(all_preds) == 0:
            continue

        # Get GT verb lemma
        gt_verbs = [extract_verb_lemma(e) for e in gt_actions]


        '''
        Find similarity scores (GTE, semantic)
        '''
        num_queries = len(gt_actions)
        all_texts = gt_actions + all_preds
        all_texts_split = [all_texts[i:i + args.chunk_size] for i in range(0, len(all_texts), args.chunk_size)]

        # Extract GTE for all GT and predictions
        all_embeddings = []
        for text_batch in all_texts_split:
            # Add instruction for embedding extractions, and set a max length for each sentence (1000 characters)
            embeddings_batch = model.encode([s[:1000] for s in text_batch], prompt="Instruct: Given a sentence, retrieve relevant passages that involve similar actions, focus particularly on the verbs\nQuery: ")
            all_embeddings.append(embeddings_batch)
            torch.cuda.empty_cache()
        all_embeddings = np.concatenate(all_embeddings, 0)

        # Find similarity score
        query_embeddings = all_embeddings[:num_queries] # GT 
        document_embeddings = all_embeddings[num_queries:] # Prediction
        pred_scores_sim = (query_embeddings @ document_embeddings.T)
        sim_scores = pred_scores_sim.max(1).tolist()


        '''
        Find verb scores (matching based)
        '''
        verb_scores = []
        for sub_idx, gt_action in enumerate(gt_actions):
            if len(pred_actions) != 0:
                regulation_scores = pred_scores_sim[sub_idx:(sub_idx+1), :len(pred_actions)][0]
                pred_scores_verb = np.array([1 if (gt_verbs[sub_idx] == pred_verb and gt_verbs[sub_idx] is not None) else 0 for pred_verb in pred_verbs])
                pred_scores_verb = pred_scores_verb * regulation_scores 
                verb_scores.append(pred_scores_verb.max())
            else:
                verb_scores.append(0.0)
        
        '''
        Combination
        '''
        action_scores = (0.8 * np.array(sim_scores) + 0.2 * np.array(verb_scores)).tolist()
        result_df.loc[row_idx, "action_score_breakdown"] = str(action_scores)
        result_df.loc[row_idx, "action_score"] = np.mean(action_scores)
        torch.cuda.empty_cache()


    # Get final scores and saving
    result_df_correct = result_df[result_df["action_score"] >= 0]
    overall_score = round(np.mean(result_df_correct["action_score"].tolist()), 5) 
    # Scaling
    overall_score = round((overall_score - 0.25) * 2, 4)
    print("The action score is " + str(overall_score))

    if args.save_path:
        result_suffix = f"_action-{str(overall_score)}"
        save_path = args.save_path.replace(".csv", f"{result_suffix}.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok = True)
        result_df.to_csv(save_path)

