import webdataset as wds

def check_webdataset_shards(shard_path):
    dataset = wds.WebDataset(shard_path).decode("rgb").to_tuple("jpg", "txt")
    print(shard_path)
    for sample in dataset:
        try:
            image, text = sample
            # 데이터가 올바르게 로드되었는지 확인
        except Exception as e:
            print(f"Error in {shard_path}: {e}")

# Example usage
check_webdataset_shards("korean-laion2B/{00000..01999}.tar")