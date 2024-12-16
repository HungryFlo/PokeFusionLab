import os
import json
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'

from datasets import load_dataset
from vllm import LLM
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_PATH = "/data/luoyx/comparison_models/Qwen2-VL-7B-Instruct"

# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype="auto", device_map="auto"
)

# default processer
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)

# 打开jsonl文件，准备写入数据，如果文件不存在则会创建它
prompts = ["Describe the image briefly.", "Describe the image.", "Describe the image. If it shows an anime character, mention the character's gender, hairstyle, clothing style, and any distinctive features or accessories."]

with open('/data/luoyx/hfy_temp/images/metadata0.jsonl', 'w') as f:
    for i in range(1, 834):  # image1-833
        print(f"Processing image{i}...")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"/data/luoyx/hfy_temp/images/image{i}.jpg",
                    },
                    {"type": "text", "text": prompts[0]},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)

        # 将结果整理为要求的格式，并写入jsonl文件
        result_dict = {
            "file_name": f"image{i}.jpg",
            "text": output_text[0] if output_text else ""
        }
        f.write(json.dumps(result_dict) + '\n')