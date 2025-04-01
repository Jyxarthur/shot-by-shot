## Film Grammar Prediction

<p align="center">
  <img src="../resources/assets/film_grammar_example.png"  width="900"/>
</p> 

#### Preliminary
Both steps utilise DINOv2 as the base model, with the checkpoints automatically downloaded. <br>
To specify the path for downloaded DINOv2 ckpts, set `model_dir` in line: `state_dict = torch.hub.load_state_dict_from_url(pretrained_weights, map_location="cpu", model_dir=#TODO)` located in `dinov2/dinov2/utils/utils.py`


#### Thread Structure Prediction
```
python thread_structure_predictor.py \ 
--dataset={dataset} \                    # e.g., "cmdad"
--anno_path={anno_path}                  # e.g., "resources/annotations/cmdad_anno_context-3.0-8.0_face-0.2-0.4.csv" \
--video_dir={video_dir} \
--save_path={save_path}
```
`--dataset`: choices are `cmdad`, `madeval`, and `tvad`. <br>
`--anno_path`: path to raw AD annotations, in the form `resources/annotations/{dataset}_anno_context-3.0-8.0_face-0.2-0.4.csv`. <br>
`--video_dir`: directory of video datasets, example file structures can be found in `resources/example_file_structures` (files are empty, for references only). <br>
`--save_path`: path to save output csv. <br>



#### Shot Scale Classification
```
python shot_scale_classifier.py \  
--dataset={dataset} \                    # e.g., "cmdad"
--anno_path={anno_path}                  # e.g., "resources/annotations/cmdad_anno_context-3.0-8.0_face-0.2-0.4.csv" \
--video_dir={video_dir} \
--resume_path={resume_path} \
--save_path={save_path}
```
`--dataset`: choices are `cmdad`, `madeval`, and `tvad`. <br>
`--anno_path`: path to raw AD annotations, in the form `resources/annotations/{dataset}_anno_context-3.0-8.0_face-0.2-0.4.csv`. <br>
`--video_dir`: directory of video datasets, example file structures can be found in `resources/example_file_structures` (files are empty, for references only). <br>
`--resume_path`: path of shot scale classifier checkpoint (on top of the original DINOv2 ckpt), which can be downloaded [here](https://drive.google.com/drive/folders/1HKqaw5aPpeTfuHkqU9Xr7TLc3va2noM1?usp=sharing). <br>
`--save_path`: path to save output csv. <br>

<span style="color:gray"><i>(The two above steps can be skipped by directly referred to the pre-computed results in the form </i> `resources/annotations/{dataset}anno_context-3.0-8.0_face-0.2-0.4_scale_thread.csv`<i>)</i></span>


## References
DINOv2: [https://github.com/facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) <br>

 
