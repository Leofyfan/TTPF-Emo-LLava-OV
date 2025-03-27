import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoProcessor, 
    LlavaOnevisionForConditionalGeneration,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model,
    PromptEncoderConfig,
    TaskType,
    PeftType,
    PeftModel,
    PeftConfig
)
from PIL import Image
import torchaudio
import numpy as np
from tqdm import tqdm
import logging
import random

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义CUFusion Adapter模块 - 模态融合和重构
class CUFusionAdapter(nn.Module):
    def __init__(self, text_dim=768, image_dim=1024, audio_dim=512, common_dim=256, unique_dim=256):
        super(CUFusionAdapter, self).__init__()
        
        # 文本模态处理
        self.text_common_encoder = nn.Linear(text_dim, common_dim)
        self.text_unique_encoder = nn.Linear(text_dim, unique_dim)
        self.text_reconstructor = nn.Linear(common_dim + unique_dim, text_dim)
        
        # 图像模态处理
        self.image_common_encoder = nn.Linear(image_dim, common_dim)
        self.image_unique_encoder = nn.Linear(image_dim, unique_dim)
        self.image_reconstructor = nn.Linear(common_dim + unique_dim, image_dim)
        
        # 音频模态处理
        self.audio_common_encoder = nn.Linear(audio_dim, common_dim)
        self.audio_unique_encoder = nn.Linear(audio_dim, unique_dim)
        self.audio_reconstructor = nn.Linear(common_dim + unique_dim, audio_dim)
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(common_dim * 3 + unique_dim * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, text_dim + image_dim)
        )
        
    def extract_features(self, text_feat, image_feat, audio_feat):
        # 分解为共性和独特特征
        text_common = self.text_common_encoder(text_feat)
        text_unique = self.text_unique_encoder(text_feat)
        
        image_common = self.image_common_encoder(image_feat)
        image_unique = self.image_unique_encoder(image_feat)
        
        audio_common = self.audio_common_encoder(audio_feat)
        audio_unique = self.audio_unique_encoder(audio_feat)
        
        return {
            'text_common': text_common, 
            'text_unique': text_unique,
            'image_common': image_common, 
            'image_unique': image_unique,
            'audio_common': audio_common, 
            'audio_unique': audio_unique
        }
    
    def reconstruct(self, features):
        # 重构文本表征（使用图像的共性和文本的独特）
        text_recon_from_image = self.text_reconstructor(
            torch.cat([features['image_common'], features['text_unique']], dim=-1)
        )
        
        # 重构图像表征（使用文本的共性和图像的独特）
        image_recon_from_text = self.image_reconstructor(
            torch.cat([features['text_common'], features['image_unique']], dim=-1)
        )
        
        # 重构文本表征（使用音频的共性和文本的独特）
        text_recon_from_audio = self.text_reconstructor(
            torch.cat([features['audio_common'], features['text_unique']], dim=-1)
        )
        
        # 重构音频表征（使用文本的共性和音频的独特）
        audio_recon_from_text = self.audio_reconstructor(
            torch.cat([features['text_common'], features['audio_unique']], dim=-1)
        )
        
        # 重构图像表征（使用音频的共性和图像的独特）
        image_recon_from_audio = self.image_reconstructor(
            torch.cat([features['audio_common'], features['image_unique']], dim=-1)
        )
        
        # 重构音频表征（使用图像的共性和音频的独特）
        audio_recon_from_image = self.audio_reconstructor(
            torch.cat([features['image_common'], features['audio_unique']], dim=-1)
        )
        
        return {
            'text_recon_from_image': text_recon_from_image,
            'image_recon_from_text': image_recon_from_text,
            'text_recon_from_audio': text_recon_from_audio,
            'audio_recon_from_text': audio_recon_from_text,
            'image_recon_from_audio': image_recon_from_audio,
            'audio_recon_from_image': audio_recon_from_image
        }
    
    def fuse_features(self, features):
        # 融合所有的共性和独特特征
        all_features = torch.cat([
            features['text_common'], features['text_unique'],
            features['image_common'], features['image_unique'],
            features['audio_common'], features['audio_unique']
        ], dim=-1)
        
        fused_features = self.fusion_layer(all_features)
        
        # 将融合特征分割回文本和图像尺寸
        text_fused, image_fused = torch.split(fused_features, [768, 1024], dim=-1)
        
        return text_fused, image_fused
    
    def forward(self, text_feat, image_feat, audio_feat):
        features = self.extract_features(text_feat, image_feat, audio_feat)
        reconstructions = self.reconstruct(features)
        text_fused, image_fused = self.fuse_features(features)
        
        return {
            'text_fused': text_fused,
            'image_fused': image_fused,
            'reconstructions': reconstructions,
            'features': features
        }


# 音频特征提取器
class AudioFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super(AudioFeatureExtractor, self).__init__()
        
        # Mel频谱图提取器
        self.mel_extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
        )
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, audio):
        # 将音频转换为单声道（如果不是）
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
            
        # 提取Mel频谱图
        mel_spec = self.mel_extractor(audio).unsqueeze(0)  # [1, 1, n_mels, time]
        
        # 使用CNN提取特征
        features = self.cnn(mel_spec)
        return features


# MELD数据集
class MELDDataset(Dataset):
    def __init__(self, data_path, processor, split='train'):
        self.data_path = data_path
        self.processor = processor
        self.split = split
        
        # 加载数据集标注
        with open(os.path.join(data_path, f'{split}_annotations.json'), 'r') as f:
            self.annotations = json.load(f)
        
        # 情感标签映射
        self.emotion_labels = {
            "neutral": 0, "joy": 1, "sadness": 2, "anger": 3, 
            "surprise": 4, "fear": 5, "disgust": 6
        }
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        
        # 加载文本
        text = item['text']
        
        # 加载图像（可能有多个视频帧）
        image_paths = item['image_paths']
        images = []
        for path in image_paths:
            try:
                img = Image.open(os.path.join(self.data_path, 'images', path)).convert('RGB')
                images.append(img)
            except Exception as e:
                logger.warning(f"无法加载图像 {path}: {e}")
                # 使用黑色图像作为替代
                img = Image.new('RGB', (224, 224), color='black')
                images.append(img)
        
        # 如果没有图像，创建一个占位图像
        if not images:
            img = Image.new('RGB', (224, 224), color='black')
            images.append(img)
        
        # 加载音频
        audio_path = item['audio_path']
        try:
            audio, sr = torchaudio.load(os.path.join(self.data_path, 'audio', audio_path))
        except Exception as e:
            logger.warning(f"无法加载音频 {audio_path}: {e}")
            # 创建一个空音频作为替代
            audio = torch.zeros(1, 16000)
        
        # 获取情感标签
        emotion = self.emotion_labels.get(item['emotion'], 0)
        
        # 为LLaVA模型准备prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"分析这个场景的情感。场景描述: {text}"},
                    *[{"type": "image"} for _ in range(len(images))]
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        return {
            'text': text,
            'images': images,
            'audio': audio,
            'emotion': emotion,
            'prompt': prompt
        }


# 结合P-Tuning和CUFusion的多模态模型
class MultimodalPTuningModel(nn.Module):
    def __init__(self, model_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", device="cuda"):
        super(MultimodalPTuningModel, self).__init__()
        
        # 加载LLaVA-OV模型
        self.llava_model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attention_2=True
        ).to(device)
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # 冻结LLaVA-OV参数
        for param in self.llava_model.parameters():
            param.requires_grad = False
        
        # 音频特征提取器
        self.audio_extractor = AudioFeatureExtractor().to(device)
        
        # CUFusion Adapter
        self.cufusion_adapter = CUFusionAdapter().to(device)
        
        # 设备
        self.device = device
        
        # 配置P-Tuning
        self._configure_peft()
        
    def _configure_peft(self):
        # 为语言模型配置P-Tuning
        peft_config = PromptEncoderConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=20,  # 虚拟token数量
            encoder_hidden_size=768,  # 编码器隐藏维度
            encoder_num_layers=2,    # 编码器层数
            prompt_tuning_init="RANDOM"  # 初始化方式
        )
        
        # 应用PEFT
        self.llava_model = get_peft_model(self.llava_model, peft_config)
        self.llava_model.print_trainable_parameters()
    
    def extract_features(self, text_prompts, images, audio):
        # 提取文本特征
        text_inputs = self.processor(text=text_prompts, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            text_features = self.llava_model.language_model.model.embed_tokens(text_inputs.input_ids)
            # 使用CLS token或平均池化
            text_features = text_features.mean(dim=1)
        
        # 提取图像特征
        image_inputs = self.processor(images=images, return_tensors="pt").to(self.device, torch.float16)
        with torch.no_grad():
            image_features = self.llava_model.vision_model(image_inputs.pixel_values).last_hidden_state
            # 使用平均池化
            image_features = image_features.mean(dim=1)
        
        # 提取音频特征
        audio = audio.to(self.device)
        audio_features = self.audio_extractor(audio)
        
        return text_features, image_features, audio_features
    
    def forward(self, batch):
        text_prompts = batch['prompt']
        images = batch['images']
        audio = batch['audio']
        
        # 批量提取特征
        batch_text_features = []
        batch_image_features = []
        batch_audio_features = []
        
        for i in range(len(text_prompts)):
            # 提取特征
            text_features, image_features, audio_features = self.extract_features(
                text_prompts[i], images[i], audio[i]
            )
            
            batch_text_features.append(text_features)
            batch_image_features.append(image_features)
            batch_audio_features.append(audio_features)
        
        # 拼接成批次张量
        batch_text_features = torch.cat(batch_text_features, dim=0)
        batch_image_features = torch.cat(batch_image_features, dim=0)
        batch_audio_features = torch.cat(batch_audio_features, dim=0)
        
        # 通过CUFusion Adapter
        outputs = self.cufusion_adapter(batch_text_features, batch_image_features, batch_audio_features)
        
        # 计算重构损失
        recons = outputs['reconstructions']
        
        # 文本重构损失
        text_image_recon_loss = F.mse_loss(recons['text_recon_from_image'], batch_text_features)
        text_audio_recon_loss = F.mse_loss(recons['text_recon_from_audio'], batch_text_features)
        
        # 图像重构损失
        image_text_recon_loss = F.mse_loss(recons['image_recon_from_text'], batch_image_features)
        image_audio_recon_loss = F.mse_loss(recons['image_recon_from_audio'], batch_image_features)
        
        # 音频重构损失
        audio_text_recon_loss = F.mse_loss(recons['audio_recon_from_text'], batch_audio_features)
        audio_image_recon_loss = F.mse_loss(recons['audio_recon_from_image'], batch_audio_features)
        
        # 总损失
        total_loss = (
            text_image_recon_loss + text_audio_recon_loss +
            image_text_recon_loss + image_audio_recon_loss +
            audio_text_recon_loss + audio_image_recon_loss
        )
        
        return {
            'loss': total_loss,
            'text_fused': outputs['text_fused'],
            'image_fused': outputs['image_fused']
        }
    
    def generate(self, text_prompt, images, audio):
        # 准备模型输入
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                    *[{"type": "image"} for _ in range(len(images))]
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # 提取特征
        text_features, image_features, audio_features = self.extract_features(prompt, images, audio)
        
        # 通过CUFusion Adapter进行融合
        with torch.no_grad():
            outputs = self.cufusion_adapter(text_features, image_features, audio_features)
            text_fused = outputs['text_fused']
            image_fused = outputs['image_fused']
        
        # 准备模型输入
        inputs = self.processor(images=images, text=prompt, return_tensors='pt').to(self.device, torch.float16)
        
        # 生成输出
        with torch.no_grad():
            output = self.llava_model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False
            )
        
        # 解码输出
        result = self.processor.decode(output[0][2:], skip_special_tokens=True)
        return result


# 自定义Trainer用于处理多模态输入
class MultimodalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss


# 训练函数
def train_multimodal_model(data_path, output_path, batch_size=4, num_epochs=5, learning_rate=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化模型
    model = MultimodalPTuningModel(device=device)
    
    # 创建数据集
    train_dataset = MELDDataset(data_path, model.processor, split='train')
    val_dataset = MELDDataset(data_path, model.processor, split='val')
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(output_path, "logs"),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        fp16=True,
        report_to="none"
    )
    
    # 定义数据整理函数
    def collate_fn(batch):
        return {
            'prompt': [item['prompt'] for item in batch],
            'images': [item['images'] for item in batch],
            'audio': torch.stack([item['audio'] for item in batch]),
            'emotion': torch.tensor([item['emotion'] for item in batch])
        }
    
    # 初始化Trainer
    trainer = MultimodalTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    model.save_pretrained(os.path.join(output_path, "final_model"))
    
    return model


# 模型推理函数
def inference(model, text_prompt, images, audio):
    return model.generate(text_prompt, images, audio)


# 保存和加载模型功能
def save_model(model, output_dir):
    # 保存PEFT模型
    model.llava_model.save_pretrained(os.path.join(output_dir, "peft_model"))
    
    # 保存CUFusion Adapter
    torch.save(model.cufusion_adapter.state_dict(), os.path.join(output_dir, "cufusion_adapter.pt"))
    
    # 保存音频特征提取器
    torch.save(model.audio_extractor.state_dict(), os.path.join(output_dir, "audio_extractor.pt"))


def load_model(model_dir, device="cuda"):
    # 加载基础模型
    model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    
    # 创建模型实例
    model = MultimodalPTuningModel(model_id=model_id, device=device)
    
    # 加载PEFT模型
    peft_config = PeftConfig.from_pretrained(os.path.join(model_dir, "peft_model"))
    model.llava_model = PeftModel.from_pretrained(
        model.llava_model, 
        os.path.join(model_dir, "peft_model")
    )
    
    # 加载CUFusion Adapter
    model.cufusion_adapter.load_state_dict(
        torch.load(os.path.join(model_dir, "cufusion_adapter.pt"))
    )
    
    # 加载音频特征提取器
    model.audio_extractor.load_state_dict(
        torch.load(os.path.join(model_dir, "audio_extractor.pt"))
    )
    
    return model


# 主函数
def main():
    # 设置随机种子
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # 数据和输出路径
    data_path = "/path/to/MELD_dataset"
    output_path = "/path/to/output"
    
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    # 训练模型
    print("开始训练模型...")
    model = train_multimodal_model(
        data_path=data_path,
        output_path=output_path,
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-4
    )
    
    # 保存模型
    print("保存模型...")
    save_model(model, output_path)
    
    # 加载模型进行推理测试
    print("加载模型进行推理测试...")
    loaded_model = load_model(output_path)
    
    # 准备测试数据
    test_text = "分析这个场景中的情感表达"
    test_image = Image.open("/path/to/test_image.jpg").convert('RGB')
    test_audio, _ = torchaudio.load("/path/to/test_audio.wav")
    
    # 进行推理
    result = inference(loaded_model, test_text, [test_image], test_audio)
    print(f"推理结果: {result}")
    
    print("完成!")


if __name__ == "__main__":
    main()