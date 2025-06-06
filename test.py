import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

model_id = "/root/autodl-tmp/llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    use_flash_attention_2=False
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

print(processor)

# # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
# # Each value in "content" has to be a list of dicts with types ("text", "image") 
# conversation = [
#     {
#       "role": "user",
#       "content": [
#           {"type": "text", "text": "What are these?"},
#           {"type": "image"},
#         ],
#     },
# ]
# prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# image_file = "http://images.cocodataset.org/val2017/000000039769.jpg"
# raw_image = Image.open(requests.get(image_file, stream=True).raw)
# inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

# output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
# print(processor.decode(output[0][2:], skip_special_tokens=True))
