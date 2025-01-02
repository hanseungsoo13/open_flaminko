'''
[x] data load
[ ] dataset, dataloader 정의
    [ ] get_dataset_fn에 get_ko_laion_dataset 정의
[ ] train 함수 분석
[ ] train
'''
import functools
import io
import json
import math
import re
import random
import numpy as np
import torch
import torchvision
import webdataset as wds
from PIL import Image
import base64
from scipy.optimize import linear_sum_assignment


from datasets import load_dataset

'''ko_laion_ds = load_dataset("Bingsu/laion2B-multi-korean-subset",
                    cache_dir="PATH/TO/CACHE/DIR",
                    )
print(ko_laion_ds)
ko_laion_ds['train'].to_csv('korean-laion2B.csv')
'''
from img2dataset import download
import shutil
import os

if __name__ == "__main__":
    output_dir = "korean-laion2B"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        processes_count=16,
        thread_count=32,
        url_list='korean-laion2B.csv',
        image_size=256,
        resize_mode = 'no',
        output_folder=output_dir,
        output_format="webdataset",
        input_format="csv",
        caption_col="TEXT",
        url_col = 'URL',
        number_sample_per_shard=1000,
        distributor="multiprocessing",
    )

