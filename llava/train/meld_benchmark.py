import json
import os
import re
import torch
import numpy as np
import random
from tqdm import tqdm
from decord import VideoReader, cpu
from PIL import Image
import copy
import argparse

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
import time


def set_seed(seed=42):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class MELDBenchmark:
    def __init__(self, 
                 base_model_path="/root/autodl-tmp/model/llava-onevision-qwen2-0.5b-ov",
                 reload_proj_path=None,
                 model_name="llava_qwen", 
                 device="cuda",
                 device_map="auto",
                 data_path="/root/autodl-tmp/data/meld_data/MELD.Raw/meld_test_standard_shuffled_part.json",
                 base_video_path="/root/autodl-tmp/data/meld_data/MELD.Raw/",
                 seed=42):
        # 设置随机种子
        set_seed(seed)
        
        self.base_model_path = base_model_path
        self.reload_proj_path = reload_proj_path
        self.model_name = model_name
        self.device = device
        self.device_map = device_map
        self.data_path = data_path
        self.base_video_path = base_video_path
        self.seed = seed
        
        self.emotions = ["anger", "disgust", "fear", "joy", "sadness", "surprise", "neutral"]
        self.sentiments = ["positive", "negative", "neutral"]
        
        self.results = []
        self.correct_emotion = 0
        self.correct_sentiment = 0
        
        # 初始化模型
        self._init_model()
        # 加载数据
        self._load_data()
    
    def _init_model(self):
        """初始化模型"""
        llava_model_args = {"multimodal": True}
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            self.base_model_path, None, self.model_name, device_map=self.device_map, 
            attn_implementation="flash_attention_2", **llava_model_args
        )
        # 加载微调后的projector权重
        if self.reload_proj_path is not None:
            print("loading projector weight...")
            projector_weights = torch.load(self.reload_proj_path, map_location="cpu")

            # 更新模型权重
            missing_keys = []
            filtered_weights = {}
            for k, v in projector_weights.items():
                # print(f"existing: {k}")
                if k in self.model.state_dict():
                    # print(f"loading {k}")
                    filtered_weights[k] = v.to(torch.float16)
                else:
                    missing_keys.append(k)
                    
            # 加载过滤后的权重
            self.model.load_state_dict(filtered_weights, strict=False, assign=True)
            print(f"successfully loading {len(filtered_weights)} weights")
            if missing_keys:
                print(f"failure loading num: {len(missing_keys)}")
            
        self.model.eval()
    
    def _load_data(self):
        """加载标准数据"""
        with open(self.data_path, "r") as f:
            self.standard_data = json.load(f)
    
    def load_video(self, video_path, max_frames_num=16):
        """提取视频帧函数"""
        if type(video_path) == str:
            vr = VideoReader(video_path, ctx=cpu(0))
        else:
            vr = VideoReader(video_path[0], ctx=cpu(0))
        total_frame_num = len(vr)
        print(f"total_frame_num: {total_frame_num}")
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
        return frames  # (frames, height, width, channels)
    
    def extract_emotion_sentiment(self, response_text):
        """提取情感和情绪的函数"""
        # 尝试匹配文本中的情感和情绪
        detected_emotion = "neutral"
        detected_sentiment = "neutral"
        
        # 情感检测
        for emotion in self.emotions:
            if re.search(r'\b' + emotion + r'\b', response_text.lower()):
                detected_emotion = emotion
                break
        
        # 情绪检测
        for sentiment in self.sentiments:
            if re.search(r'\b' + sentiment + r'\b', response_text.lower()):
                detected_sentiment = sentiment
                break
        
        return detected_emotion, detected_sentiment
    
    def process_single_video(self, video_path, prompt=None):
        """处理单个视频并返回情感和情绪预测"""
        try:
            # 加载视频
            video_frames = self.load_video(video_path)
            
            # 处理视频帧
            frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
            image_tensors = [frames]
            
            # 准备会话输入
            conv_template = "qwen_1_5"
            if prompt is None:
                prompt = f"{DEFAULT_IMAGE_TOKEN}\nJudge the emotion of character and the sentiment of the scene in the video. For emotion, only choose one from: anger, disgust, fear, joy, neutral, sadness, surprise. For sentiment, only choose one from: positive, neutral, negative."
            else:
                prompt = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
            
            conv = copy.deepcopy(conv_templates[conv_template])
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt_question = conv.get_prompt()
            
            input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            image_sizes = [frame.size for frame in video_frames]
            
            # 设置生成时的随机种子
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            
            # 生成回答
            with torch.no_grad():
                cont = self.model.generate(
                    input_ids,
                    images=image_tensors,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=4096,
                    modalities=["video"],
                )
            
            text_output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
            
            # 提取情感和情绪
            pred_emotion, pred_sentiment = self.extract_emotion_sentiment(text_output)
            
            return {
                "model_response": text_output,
                "emotion": pred_emotion,
                "sentiment": pred_sentiment
            }
            
        except Exception as e:
            print(f"处理视频时出错: {str(e)}")
            return {
                "model_response": "",
                "emotion": "neutral",
                "sentiment": "neutral",
                "error": str(e)
            }
    
    def evaluate(self):
        """评估所有测试数据"""
        self.results = []
        self.correct_emotion = 0
        self.correct_sentiment = 0
        
        for sample in tqdm(self.standard_data):
            sample_id = sample["id"]
            video_path = os.path.join(self.base_video_path, sample["video"])
            
            try:
                # 加载视频
                start_load_video = time.time()
                video_frames = self.load_video(video_path)
                end_load_video = time.time()
                print(f"load video using {end_load_video - start_load_video} s")
                
                # 处理视频帧
                start_image_processor = time.time()

                frames = self.image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half().cuda()
                image_tensors = [frames]
                
                end_image_processor = time.time()
                print(f"image_processor video using {end_image_processor - start_image_processor} s")
                
                # 准备会话输入
                conv_template = "qwen_1_5"
                question = f"{DEFAULT_IMAGE_TOKEN}\nJudge the emotion of character and the sentiment of the scene in the video. For emotion, only choose one from: anger, disgust, fear, joy, neutral, sadness, surprise. For sentiment, only choose one from: positive, neutral, negative."
                
                conv = copy.deepcopy(conv_templates[conv_template])
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
                image_sizes = [frame.size for frame in video_frames]
                
                # 设置生成时的随机种子
                torch.manual_seed(self.seed)
                torch.cuda.manual_seed(self.seed)
                
                start_generate = time.time()

                
                # 生成回答
                with torch.no_grad():
                    cont = self.model.generate(
                        input_ids,
                        images=image_tensors,
                        image_sizes=image_sizes,
                        do_sample=False,
                        temperature=0,
                        max_new_tokens=4096,
                        modalities=["video"],
                    )
                    
                end_generate = time.time()
                print(f"generate video using {end_generate - start_generate} s")
                
                text_output = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
                
                # 提取情感和情绪
                pred_emotion, pred_sentiment = self.extract_emotion_sentiment(text_output)
                
                # 检查预测是否正确
                is_emotion_correct = pred_emotion == sample["emotion"]
                is_sentiment_correct = pred_sentiment == sample["sentiment"]
                
                if is_emotion_correct:
                    self.correct_emotion += 1
                if is_sentiment_correct:
                    self.correct_sentiment += 1
                
                # 添加结果
                result = {
                    "id": sample_id,
                    "video": sample["video"],
                    "standard_emotion": sample["emotion"],
                    "standard_sentiment": sample["sentiment"],
                    "model_response": text_output,
                    "extracted_emotion": pred_emotion,
                    "extracted_sentiment": pred_sentiment,
                    "emotion_correct": is_emotion_correct,
                    "sentiment_correct": is_sentiment_correct
                }
                
                self.results.append(result)
            
            except Exception as e:
                print(f"处理样本 {sample_id} 时出错: {str(e)}")
                continue
        
        return self.get_summary()
    
    def get_summary(self):
        """获取评估摘要"""
        total_samples = len(self.results)
        emotion_accuracy = self.correct_emotion / total_samples if total_samples > 0 else 0
        sentiment_accuracy = self.correct_sentiment / total_samples if total_samples > 0 else 0
        
        # 添加准确率信息
        summary = {
            "total_samples": total_samples,
            "emotion_accuracy": emotion_accuracy,
            "sentiment_accuracy": sentiment_accuracy
        }
        
        return summary
    
    def save_results(self, output_file):
        """保存结果到文件"""
        summary = self.get_summary()
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": self.results, "summary": summary}, f, indent=2, ensure_ascii=False)
        
        print(f"\n评估完成! 结果已保存到: {output_file}")
        print(f"情感(emotion)准确率: {summary['emotion_accuracy']:.4f}")
        print(f"情绪(sentiment)准确率: {summary['sentiment_accuracy']:.4f}")


# 使用示例
if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='MELD基准测试评估')
    parser.add_argument('--reload_proj_path', type=str, default=None, 
                        help='微调后的projector权重路径')
    parser.add_argument('--base_model_path', type=str, required=True, 
                        help='基础模型路径')
    parser.add_argument('--output_file', type=str, 
                        default="/root/project/llava/TTPF-Emo-LLava-OV/eval/results/meld_test_results.json",
                        help='结果输出文件路径')
    args = parser.parse_args()
    
    # 设置全局随机种子
    set_seed(42)
    
    benchmark = MELDBenchmark(seed=42, reload_proj_path=args.reload_proj_path, base_model_path=args.base_model_path)
    benchmark.evaluate()
    benchmark.save_results(output_file=args.output_file)