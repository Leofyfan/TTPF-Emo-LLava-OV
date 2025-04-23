import requests
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
from decord import VideoReader, cpu
import numpy as np

model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
    use_flash_attention_2=True
).to(0)

processor = AutoProcessor.from_pretrained(model_id)

# Function to extract frames from video
def load_video(video_path, max_frames_num):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

# Load and process video
video_path = "meld_data/dia1_utt0.mp4"
video_frames = load_video(video_path, 16)

# Define a chat history and use `apply_chat_template` to get correctly formatted prompt
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Give the emotion of the video(emotion classes must belong to [neutral, joy, sadness, anger, surprise, fear, disgust]), and give a score between 0 and 100. The answer format should be like this: {'score': 50, 'emotion': 'happy'}. Do not give any other information!"},
          {"type": "video"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Prepare input for the model
inputs = processor(text=prompt, videos=video_frames, return_tensors='pt').to(0, torch.float16)

# Generate response
output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0][2:], skip_special_tokens=True))