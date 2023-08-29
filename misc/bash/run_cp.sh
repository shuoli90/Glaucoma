#! /bin/bash

python cp_df.py --json_file "informative_images.json" > cp_info
python cp_df.py --json_file "informative_images_first.json" > cp_info_first
python cp_df.py --json_file "full_images.json" > cp_full
python cp_df.py --json_file "full_images_first.json" > cp_full_first
python cp_df.py --json_file "binary_select_images.json" > cp_binary
python cp_df.py --json_file "segformer_select_images.json" > cp_select
