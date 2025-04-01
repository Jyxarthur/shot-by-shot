import sys
import numpy as np


class PromptLoader():
    def __init__(self, prompt_idx, video_type, label_type):
        self.prompt_idx = prompt_idx
        self.video_type = video_type
        self.label_type = label_type

    
    def apply(self, char_text, current_shots=None, threads=None, shot_scales=None):       
        # For scales for current shots and context (past and future) shots
        current_shot_scales = [e for i, e in enumerate(shot_scales) if i in current_shots]
        context_shot_scales = [e for i, e in enumerate(shot_scales) if i not in current_shots]

        # Take average of current shots to find effective shot scale
        if len(current_shot_scales) == 0 and len(context_shot_scales) == 0:
            eff_shot_scale = 2
        elif len(current_shot_scales) == 0:
            eff_shot_scale = np.mean(context_shot_scales)
        else:
            eff_shot_scale = np.mean(current_shot_scales)

        # Formulate Stage I factor based on effective shot scale
        if self.prompt_idx == 0:
            if eff_shot_scale < 1:
                prompt_idx = 1
            elif eff_shot_scale >= 3.5:
                prompt_idx = 2
            elif eff_shot_scale >= 1.5 and eff_shot_scale < 3.:
                prompt_idx = 3
            else:
                prompt_idx = 4
        else:
            prompt_idx = self.prompt_idx

        # Formulate text that contains current shot indices
        current_shot_texts = []
        for current_shot in current_shots:
            current_shot_texts.append(f"Shot {current_shot}") 
        current_shot_text = "[" + ", ".join(current_shot_texts) + "]"
        

        # Different prompts, each with a single (or none) additional factor
        if prompt_idx == 1: # facial expression
            if len(threads) == 0: # no thread structure
                thread_text = ""
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: '';\n4. Facial expressions: ''."
            else:
                thread_texts = []
                for thread in threads:
                    shot_texts = []
                    for thread_single in thread:
                        shot_texts.append(f"Shot {thread_single}") 
                    shot_text = "[" + ", ".join(shot_texts) + "]"
                    thread_texts.append(f"{shot_text} share the same camera setup")
                thread_text = ", and ".join(thread_texts) + ". "
                thread_text = f"Finally, in one sentence, briefly explain why {thread_text}\n"
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: '';\n4. Facial expressions: ''.\n\nExplanation: ''."
            
            # Inject information into the prompt
            general_prompt = (
                "Please watch the following {video_type} clip, where different shot numbers are labelled on the top-left of each frame.\n"
                f"Please briefly describe what happened in {current_shot_text} in the four steps below:\n"
                "1. Identify main characters (if {label_type} are available){char_text};\n"
                "2. Describe the actions of characters, i.e., who is doing what, focusing on the movements;\n" 
                "3. Describe the interactions between characters, such as looking;\n"
                "4. Describe the facial expressions of characters.\n"
                f"{thread_text}"
                "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "   
                "Make sure you do not hallucinate information.\n"
                f"{template}"
            ) 
            general_prompt = general_prompt.format(video_type=self.video_type, char_text=char_text, label_type=self.label_type)

        elif prompt_idx == 2: # environment
            if len(threads) == 0: # no thread structure
                thread_text = ""
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: '';\n4. Environment: ''."
            else:
                thread_texts = []
                for thread in threads:
                    shot_texts = []
                    for thread_single in thread:
                        shot_texts.append(f"Shot {thread_single}") 
                    shot_text = "[" + ", ".join(shot_texts) + "]"
                    thread_texts.append(f"{shot_text} share the same camera setup")
                thread_text = ", and ".join(thread_texts) + ". "
                thread_text = f"Finally, in one sentence, briefly explain why {thread_text}\n"
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: '';\n4. Environment: ''.\n\nExplanation: ''."

            # Inject information into the prompt
            general_prompt = (
                "Please watch the following {video_type} clip, where different shot numbers are labelled on the top-left of each frame.\n"
                f"Please briefly describe what happened in {current_shot_text} in the four steps below:\n"
                "1. Identify main characters (if {label_type} are available){char_text};\n"
                "2. Describe the actions of characters, i.e., who is doing what, focusing on the movements;\n" 
                "3. Describe the interactions between characters, such as looking;\n"
                "4. Describe the environment, focusing on the location, furniture, entrances and exits, etc.\n"
                f"{thread_text}"
                "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "
                "Make sure you do not hallucinate information.\n"   
                f"{template}"
            ) 
            general_prompt = general_prompt.format(video_type=self.video_type, char_text=char_text, label_type=self.label_type)


        elif prompt_idx == 3: # key objects
            if len(threads) == 0: # no thread structure
                thread_text = ""
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: '';\n4. Key objects: ''."
            else:
                thread_texts = []
                for thread in threads:
                    shot_texts = []
                    for thread_single in thread:
                        shot_texts.append(f"Shot {thread_single}") 
                    shot_text = "[" + ", ".join(shot_texts) + "]"
                    thread_texts.append(f"{shot_text} share the same camera setup")
                thread_text = ", and ".join(thread_texts) + ". "
                thread_text = f"Finally, in one sentence, briefly explain why {thread_text}\n"
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: '';\n4. Key objects: ''.\n\nExplanation: ''."

            # Inject information into the prompt
            general_prompt = (
                "Please watch the following {video_type} clip, where different shot numbers are labelled on the top-left of each frame.\n"
                f"Please briefly describe what happened in {current_shot_text} in the four steps below:\n"
                "1. Identify main characters (if {label_type} are available){char_text};\n"
                "2. Describe the actions of characters, i.e., who is doing what, focusing on the movements;\n" 
                "3. Describe the interactions between characters, such as looking;\n"
                "4. Describe the key objects that characters interact with.\n"
                f"{thread_text}"
                "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "   
                "Make sure you do not hallucinate information.\n"
                f"{template}"
            ) 
            general_prompt = general_prompt.format(video_type=self.video_type, char_text=char_text, label_type=self.label_type)


        elif prompt_idx == 4: # None
            if len(threads) == 0: # no thread structure
                thread_text = ""
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: ''."
            else:
                thread_texts = []
                for thread in threads:
                    shot_texts = []
                    for thread_single in thread:
                        shot_texts.append(f"Shot {thread_single}") 
                    shot_text = "[" + ", ".join(shot_texts) + "]"
                    thread_texts.append(f"{shot_text} share the same camera setup")
                thread_text = ", and ".join(thread_texts) + ". "
                thread_text = f"Finally, in one sentence, briefly explain why {thread_text}\n"
                template = "### Answer Template ###\nDescription:\n1. Main characters: '';\n2. Actions: '';\n3. Character-character interactions: ''.\n\nExplanation: ''."

            # Inject information into the prompt
            general_prompt = (
                "Please watch the following {video_type} clip, where different shot numbers are labelled on the top-left of each frame.\n"
                f"Please briefly describe what happened in {current_shot_text} in the three steps below:\n"
                "1. Identify main characters (if {label_type} are available){char_text};\n"
                "2. Describe the actions of characters, i.e., who is doing what, focusing on the movements;\n"
                "3. Describe the interactions between characters, such as looking.\n"
                f"{thread_text}"
                "Note, colored {label_type} are provided for character indications only, DO NOT mention them in the description. "   
                "Make sure you do not hallucinate information.\n"
                f"{template}"
            ) 
            general_prompt = general_prompt.format(video_type=self.video_type, char_text=char_text, label_type=self.label_type)
        
        else:
            print("Check prompt_idx")
            sys.exit(0)
      
        return general_prompt