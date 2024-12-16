from datasets import load_dataset

# 从 Hugging Face Hub 加载数据集
def read_dataset_from_hub(dataset_name):
    try:
        # 使用datasets库加载数据集
        dataset = load_dataset(dataset_name)
        return dataset
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

dataset_name = 'svjack/pokemon-blip-captions-en-zh'
dataset = read_dataset_from_hub(dataset_name)

index = 0
for i in dataset['train']:
    index += 1
    # print(type(i['image']))
    i['image'].save(f"./image/all_images/image{index}.jpg", 'JPEG')
