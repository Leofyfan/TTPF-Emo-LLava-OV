# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle

# from PIL import Image
# import requests
# import copy
# import torch

# import sys
# import warnings

# warnings.filterwarnings("ignore")
# pretrained = "/root/autodl-tmp/model/llava-onevision-qwen2-0.5b-ov"
# model_name = "llava_qwen"
# device = "cuda"
# device_map = "auto"
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa")  # Add any other thing you want to pass in llava_model_args

# model.eval()

# url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
# image = Image.open(requests.get(url, stream=True).raw)
# image_tensor = process_images([image], image_processor, model.config)
# image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

# conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
# question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
# conv = copy.deepcopy(conv_templates[conv_template])
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt_question = conv.get_prompt()

# input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# image_sizes = [image.size]


# cont = model.generate(
#     input_ids,
#     images=image_tensor,
#     image_sizes=image_sizes,
#     do_sample=False,
#     temperature=0,
#     max_new_tokens=4096,
# )
# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print(text_outputs)




########################################## video example #############################################################33333



# from operator import attrgetter
# from llava.model.builder import load_pretrained_model
# from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle

# import torch
# import cv2
# import numpy as np
# from PIL import Image
# import requests
# import copy
# import warnings
# from decord import VideoReader, cpu

# warnings.filterwarnings("ignore")
# # Load the OneVision model
# pretrained = "/root/autodl-tmp/model/llava-onevision-qwen2-0.5b-ov"
# model_name = "llava_qwen"
# device = "cuda"
# device_map = "auto"
# llava_model_args = {
#     "multimodal": True,
# }
# tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map, attn_implementation="sdpa", **llava_model_args)

# model.eval()


# # Function to extract frames from video
# def load_video(video_path, max_frames_num):
#     if type(video_path) == str:
#         vr = VideoReader(video_path, ctx=cpu(0))
#     else:
#         vr = VideoReader(video_path[0], ctx=cpu(0))
#     total_frame_num = len(vr)
#     uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
#     frame_idx = uniform_sampled_frames.tolist()
#     spare_frames = vr.get_batch(frame_idx).asnumpy()
#     return spare_frames  # (frames, height, width, channels)


# # Load and process video
# video_path = "/root/autodl-tmp/data/meld_data/MELD.Raw/train_splits/dia799_utt1.mp4"
# video_frames = load_video(video_path, 16)
# print(video_frames.shape) # (16, 1024, 576, 3)
# image_tensors = []
# frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
# image_tensors.append(frames)

# # Prepare conversation input
# conv_template = "qwen_1_5"
# question = f"{DEFAULT_IMAGE_TOKEN}\nJudge the emotion of the people in the video."

# conv = copy.deepcopy(conv_templates[conv_template])
# conv.append_message(conv.roles[0], question)
# conv.append_message(conv.roles[1], None)
# prompt_question = conv.get_prompt()

# input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
# image_sizes = [frame.size for frame in video_frames]

# # Generate response
# cont = model.generate(
#     input_ids,
#     images=image_tensors,
#     image_sizes=image_sizes,
#     do_sample=False,
#     temperature=0,
#     max_new_tokens=4096,
#     modalities=["video"],
# )
# text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
# print(text_outputs[0])



from operator import attrgetter
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

import torch
import cv2
import numpy as np
from PIL import Image
import requests
import copy
import warnings
from decord import VideoReader, cpu
import os
import random
def set_seed(seed=724):
    """random seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(724)

warnings.filterwarnings("ignore")

# 路径设置
base_model_path = "/root/autodl-tmp/model/llava-onevision-qwen2-0.5b-ov"
projector_path = "/root/autodl-tmp/model/mnt/llava-ov-checkpoints/llavaov_finetune_meld_data_20250408_123148/mm_projector.bin"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

# 先加载基本模型
llava_model_args = {
    "multimodal": True,
}
tokenizer, model, image_processor, max_length = load_pretrained_model(
    base_model_path, None, model_name, device_map=device_map, 
    attn_implementation="sdpa", **llava_model_args
)

# 加载微调后的projector权重
print("loading projector weight...")
projector_weights = torch.load(projector_path, map_location="cpu")

# 更新模型权重
missing_keys = []
filtered_weights = {}
for k, v in projector_weights.items():
    # print(f"已有的：{k}")
    if k in model.state_dict():
        # print(k)
        filtered_weights[k] = v.to(torch.float16)
    else:
        missing_keys.append(k)
        

# print(filtered_weights)

# 加载过滤后的权重
model.load_state_dict(filtered_weights, strict=False)


print(f"successfully loading {len(filtered_weights)} weights")
if missing_keys:
    print(f"failure loading num: {len(missing_keys)}")

model.eval()

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

# Load and process video
video_path = "/root/autodl-tmp/data/meld_data/MELD.Raw/train_splits/dia860_utt0.mp4"
video_frames = load_video(video_path, 16)
print(f"video frames: {video_frames.shape}") # (16, 1024, 576, 3)
image_tensors = []
frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
image_tensors.append(frames)

# Prepare conversation input
conv_template = "qwen_1_5"
question = f"{DEFAULT_IMAGE_TOKEN}\nJudge the emotion of character and the sentiment of the scene in the video. For emotion, only choose one from: anger, disgust, fear, joy, neutral, sadness, surprise. For sentiment, only choose one from: positive, neutral, negative."

conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [frame.size for frame in video_frames]

print("generating...")
# Generate response
cont = model.generate(
    input_ids,
    images=image_tensors,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
    modalities=["video"],
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print("answering:")
print(text_outputs[0])