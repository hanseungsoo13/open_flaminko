from datasets import load_dataset
import pandas as pd

mmc4_ds = load_dataset("silk-road/MMC4-130k-image-english",
                    cache_dir="PATH/TO/CACHE/DIR")
mmc4_df = pd.DataFrame(mmc4_ds['train'])[:50000]
mmc4_df.to_csv('english-mmc4-50k.csv')

from img2dataset import download
import shutil
import os

if __name__ == "__main__":
    output_dir = "english-mmc4-50k"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    download(
        processes_count=16,
        thread_count=32,
        url_list='english-mmc4-50k.csv',
        image_size=256,
        resize_mode = 'no',
        output_folder=output_dir,
        output_format="webdataset",
        input_format="csv",
        caption_col="caption",
        url_col = 'url',
        number_sample_per_shard=1000,
        distributor="multiprocessing",
    )

