from num2words import num2words
import numpy as np
import copy



def get_user_prompt(mode, prompt_idx, verb_list, text_pred, word_limit, examples):
    text_pred = f"\"{text_pred.strip()}\""

    # Format examples
    example_sentence = ""
    for selected_example in examples:
        example_sentence += "{\"summarized_AD\": \""+ f"{selected_example}" + "\"}\n"

    if mode == "single":
        if prompt_idx == 0:
            template = "{\"summarized_AD\": \"\"}"
            user_prompt = (
                "Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
                f"Description: {text_pred}\n\n"
                "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
                "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                "Do not mention characters' mood.\n"
                "Do not hallucinate information that is not mentioned in the input.\n"
                f"Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
                "Provide the AD from a narrator perspective.\n"
                f"Limit the length of the output within {word_limit} words.\n\n"
                f"Output template (in JSON format): {template}.\n"
                "Here are some example outputs:\n"
                f"{example_sentence}"
            )
    else: # assistant mode
        if prompt_idx == 0:
            template = "{\"summarized_AD_1\": \"\",\n\"summarized_AD_2\": \"\",\n\"summarized_AD_3\": \"\",\n\"summarized_AD_4\": \"\",\n\"summarized_AD_5\": \"\"}"
            user_prompt = (
                "Please summarize the following description for one movie clip into ONE succinct audio description (AD) sentence.\n"
                f"Description: {text_pred}\n\n"
                "Focus on the most attractive characters, their actions, and related key objects (focus on point 2., supplemented by point 3.).\n"
                "For characters, use their first names, remove titles such as 'Mr.' and 'Dr.'. If names are not available, use pronouns such as 'He' and 'her', do not use expression such as 'a man'.\n"
                "For actions, avoid mentioning the camera, and do not focus on 'talking'.\n"
                "For objects, especially when no characters are involved, prioritize describing concrete and specific ones.\n"
                "Do not mention characters' mood.\n"
                "Do not hallucinate information that is not mentioned in the input.\n"
                f"Try to identify the following motions (with decreasing priorities): {verb_list}, and use them in the description.\n"
                "Provide 5 possible ADs from a narrator perspective, each offering a valid and distinct summary by emphasizing different key characters, actions, and movements present in the scene.\n"
                f"Limit the length of each output within {word_limit} words.\n\n"
                f"Output template (in JSON format): {template}.\n"
                "Here are some example outputs:\n"
                f"{example_sentence}"
            )
    
    return user_prompt


