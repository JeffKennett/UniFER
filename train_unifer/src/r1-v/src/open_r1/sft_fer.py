"""
UniFER 监督微调 (SFT) 训练脚本
=================================

功能说明:
    本脚本实现UniFER模型的第一阶段训练 - 监督微调(Supervised Fine-Tuning)。
    使用UniFER-CoT-230K数据集，让基础模型Qwen2.5-VL学会:
    1. 面部表情识别任务
    2. 思维链(Chain-of-Thought)推理
    3. 结构化输出 (<think></think><answer></answer>格式)

训练流程:
    1. 加载预训练的Qwen2.5-VL-7B-Instruct模型
    2. 准备训练数据（图像 + 问题 + CoT响应）
    3. 使用DeepSpeed进行分布式训练
    4. 保存微调后的模型检查点

技术栈:
    - 框架: Transformers + TRL
    - 分布式: DeepSpeed ZeRO-2
    - 优化: AdamW + Cosine调度
    - 加速: Flash Attention 2 + 梯度检查点

作者: UniFER团队
日期: 2025
"""

import os
import json
import random
import requests
import torch
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration
)
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
)
from accelerate import Accelerator
from qwen_vl_utils import process_vision_info

from datasets import Dataset, DatasetDict

import wandb

from typing import List, Dict, Any, Optional

@dataclass
class SFTcriptArguments(ScriptArguments):
    """
    SFT训练脚本的自定义参数
    
    扩展基础ScriptArguments，添加UniFER特定的配置参数。
    """

    data_dir: str = field(
        default="/dataset/",
        metadata={"help": "数据集图像文件的存储目录路径"},
    )
    


def get_current_device():
    """
    获取当前设备
    
    返回值:
        对于GPU: 返回当前进程的本地索引（支持多GPU训练）
        对于CPU: 返回"cpu"
    
    用途:
        - 多GPU训练时，每个进程分配到不同的GPU
        - Accelerate库自动处理设备分配
    """
def get_current_device():
    """
    获取当前设备
    
    返回值:
        对于GPU: 返回当前进程的本地索引（支持多GPU训练）
        对于CPU: 返回"cpu"
    
    用途:
        - 多GPU训练时，每个进程分配到不同的GPU
        - Accelerate库自动处理设备分配
    """
    return Accelerator().local_process_index if torch.cuda.is_available() else "cpu"


def prepare_dataset(script_args, example: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    准备单个训练样本
    
    参数:
        script_args: 脚本参数，包含data_dir等配置
        example: 原始数据样本，包含image_path, question, response
    
    返回值:
        格式化的消息字典，符合对话模板要求
    
    数据转换:
        原始格式: {"image_path": "...", "question": "...", "response": "..."}
        输出格式: {"messages": [
            {"role": "system", "content": [...]},
            {"role": "user", "content": [图像, 文本]},
            {"role": "assistant", "content": [响应]}
        ]}
    
    技术要点:
        - 系统提示词为空（使用默认行为）
        - 图像统一resize到224x224
        - 用户消息包含图像和问题
        - 助手消息包含CoT格式的响应
    """
    system_message = ""  # 系统提示词（可自定义模型角色）
    
    # 构造多轮对话格式
    messages = [
        # 系统消息（定义模型行为）
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}]
        },
        # 用户消息（图像 + 问题）
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": script_args.data_dir + example['image_path'],  # 完整图像路径
                    "resized_height": 224,  # 统一高度
                    "resized_width":  224,  # 统一宽度
                },
                {
                    "type": "text",
                    "text": example['question']  # 表情识别问题
                }
            ]
        },
        # 助手消息（CoT格式的答案）
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example['response']}]
            # response格式: "<think>推理过程</think><answer>标签</answer>"
        }
    ]
    
    return {"messages": messages}

def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    批量数据整理函数 (Data Collator)
    
    参数:
        examples: 批量样本列表，每个样本包含messages字段
    
    返回值:
        处理后的batch字典，包含input_ids, attention_mask, pixel_values, labels等
    
    处理流程:
        1. 应用对话模板，将messages转为文本
        2. 处理图像，提取视觉特征
        3. 使用processor统一编码文本和图像
        4. 构造训练标签（labels）
        5. 屏蔽特殊token和视觉token的损失
    
    技术要点:
        - padding确保batch内样本长度一致
        - pad_token位置的label设为-100（计算loss时忽略）
        - 视觉token的label也设为-100（只在文本token上计算loss）
    """
    texts = []          # 存储处理后的文本
    # video_inputs = []   # （预留）视频输入
    # image_inputs = []   # 图像输入（在循环中处理）

    # ========== 处理每个样本 ==========
    for i, example in enumerate(examples):
        try:
            # 应用对话模板：将messages转换为模型输入格式的文本
            # 例如: "<|im_start|>user\n图像\n问题<|im_end|>\n<|im_start|>assistant\n回答<|im_end|>"
            texts.append(processor.apply_chat_template(example["messages"], tokenize=False))
            
            # 处理视觉信息：加载图像并预处理
            # 返回: (image_inputs, video_inputs)
            image_inputs, _ = process_vision_info(example["messages"])
            
            # 调试信息（已注释）
            # print("prompts:", examples)
            # print("image_inputs:", len(image_inputs),image_inputs)
            # print("prompts_text:", len(texts),texts)
        
        except Exception as e:
            # 错误处理：提供详细的错误信息
            raise ValueError(f"处理样本 {i} 失败: {e}")

    # ========== 统一编码文本和图像 ==========
    # processor同时处理文本和图像，返回模型输入
    inputs = processor(
        text=texts,                 # 文本列表
        images=image_inputs,        # 图像列表
        return_tensors="pt",        # 返回PyTorch张量
        padding=True                # padding到batch内最大长度
    )

    # ========== 构造训练标签 ==========
    # 复制input_ids作为labels
    labels = inputs["input_ids"].clone()
    
    # 将padding token的label设为-100（CrossEntropyLoss会忽略）
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # ========== 处理视觉token ==========
    # 根据processor类型确定视觉token的ID
    # Qwen2VL使用特定的视觉token ID，其他模型使用<image> token
    visual_tokens = [151652, 151653, 151656] if isinstance(processor, Qwen2VLProcessor) else [
        processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    ]

    # 将视觉token的label也设为-100
    # 原因: 视觉token不应该参与语言建模loss的计算
    for visual_token_id in visual_tokens:
        labels[labels == visual_token_id] = -100

    # 将labels添加到inputs中
    inputs["labels"] = labels
    return inputs

if __name__ == "__main__":
    """
    主训练流程
    
    执行步骤:
        1. 解析命令行参数（训练配置、模型配置等）
        2. 加载训练数据集
        3. 初始化模型和处理器
        4. 准备数据（转换为对话格式）
        5. 初始化Trainer并开始训练
        6. 保存训练后的模型
    """
    
    # ========== 1. 解析参数 ==========
    # TrlParser同时解析三组参数: 脚本参数、训练参数、模型参数
    parser = TrlParser((SFTcriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    # ========== 2. 配置训练参数 ==========
    # 梯度检查点配置: use_reentrant=False 使用非重入模式（推荐）
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    
    # 不移除未使用的列（保留所有数据字段）
    training_args.remove_unused_columns = False
    
    # 跳过数据集预处理（我们手动处理）
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # ========== 3. 加载数据集 ==========
    # 支持本地JSON文件或HuggingFace Hub数据集
    if script_args.dataset_name.endswith('.json') or script_args.dataset_name.endswith('.jsonl'):
        # 从本地JSON文件加载
        dataset = DatasetDict({"train": Dataset.from_json(script_args.dataset_name)})
    else:
        # 从HuggingFace Hub加载
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # ========== 4. 设置模型参数 ==========
    # 确定数据类型（float32, bfloat16等）
    torch_dtype = (
        model_config.torch_dtype
        if model_config.torch_dtype in ["auto", None]
        else getattr(torch, model_config.torch_dtype)
    )

    # （可选）4-bit量化配置 - 节省显存但可能影响性能
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,                          # 启用4-bit量化
    #     bnb_4bit_use_double_quant=True,             # 双重量化
    #     bnb_4bit_quant_type="nf4",                  # NormalFloat4量化类型
    #     bnb_4bit_compute_dtype=torch.bfloat16       # 计算时使用bfloat16
    # )

    # ========== 5. 模型初始化参数 ==========
    model_kwargs = dict(
        revision=model_config.model_revision,               # 模型版本
        trust_remote_code=model_config.trust_remote_code,   # 是否信任远程代码
        torch_dtype=torch_dtype,                            # 数据类型
        device_map=get_kbit_device_map(),                   # 设备映射（多GPU）
        # quantization_config=bnb_config,                   # （可选）量化配置
    )
    
    # ========== 6. 加载模型 ==========
    # 根据模型名称选择对应的类
    if "Qwen2-VL" in model_config.model_name_or_path:
        # Qwen2-VL系列
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    elif "Qwen2.5-VL" in model_config.model_name_or_path:
        # Qwen2.5-VL系列（推荐）
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    else:
        # 其他视觉-语言模型
        model = AutoModelForVision2Seq.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )

    # ========== 7. 加载处理器 ==========
    # Processor包含tokenizer和image_processor
    processor = AutoProcessor.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code
    )

    # ========== 8. 准备训练数据 ==========
    # 将原始数据转换为对话格式
    print(f"正在准备数据集，共 {len(dataset['train'])} 个样本...")
    prepared_dataset = [
        prepare_dataset(script_args, example) 
        for example in dataset['train']
    ]
    print("数据集准备完成!")

    # ========== 9. 初始化Weights & Biases（可选） ==========
    # 用于实时监控训练进度
    if training_args.report_to == "wandb":
        wandb.init(project="fer-vlm-training")

    # ========== 10. 初始化Trainer ==========
    # SFTTrainer是专门用于监督微调的Trainer
    trainer = SFTTrainer(
        model=model,                                    # 要训练的模型
        args=training_args,                             # 训练参数
        train_dataset=prepared_dataset,                 # 训练数据
        data_collator=collate_fn,                       # 数据整理函数
        peft_config=get_peft_config(model_config),     # PEFT配置（如LoRA）
        # tokenizer=processor.tokenizer                # （可选）tokenizer
    )

    # ========== 11. 开始训练 ==========
    print("开始训练...")
    trainer.train()
    print("训练完成!")

    # ========== 12. 保存模型 ==========
    print(f"正在保存模型到 {training_args.output_dir}...")
    
    # 保存模型权重
    trainer.save_model(training_args.output_dir)
    
    # 保存处理器（tokenizer + image_processor）
    processor.save_pretrained(training_args.output_dir)

    # ========== 13. 恢复配置（仅主进程） ==========
    if trainer.accelerator.is_main_process:
        # 恢复KV缓存设置，用于快速推理
        # 训练时通常关闭缓存以节省显存
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    # ========== 14. 清理资源 ==========
    print("清理资源...")
    del model
    del trainer
    torch.cuda.empty_cache()  # 清空GPU缓存
    
    # 结束wandb记录
    if training_args.report_to == "wandb":
        wandb.finish()
    
    print("训练流程全部完成!")
