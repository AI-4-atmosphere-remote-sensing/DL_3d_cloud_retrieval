#!/bin/bash

folder_path=$1
echo $folder_path
python raw_data_preprocess.py --data_dir $folder_path
python generate_patch_dataset.py

