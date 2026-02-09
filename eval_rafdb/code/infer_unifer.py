"""
UniFER RAF-DB 数据集推理脚本
=================================

功能说明:
    本脚本用于在RAF-DB数据集上进行面部表情识别的推理。
    使用UniFER-7B模型对测试集中的图像进行表情预测。

主要步骤:
    1. 加载预训练的UniFER-7B模型和处理器
    2. 读取RAF-DB测试数据集
    3. 对每张图像进行推理，生成表情识别结果
    4. 保存推理结果到JSON文件

作者: UniFER团队
日期: 2025
"""

import json
import torch
from tqdm import tqdm
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==================== 配置参数 ====================
# 输入文件: RAF-DB测试集的问答数据
input_file = "./UniFER/eval_rafdb/data/rafdb_qa.json"

# 输出文件: 保存推理结果
output_file = "./UniFER/eval_rafdb/results/rafdb_unifer_7b_results.json"

# 模型路径: UniFER-7B模型的本地路径
model_name = "./UniFER/model/UniFER-7B"

def load_model():
    """
    加载UniFER-7B模型和处理器
    
    功能说明:
        从指定路径加载预训练的Qwen2.5-VL多模态大语言模型及其对应的处理器。
        模型使用float32精度以保证推理质量，支持自动设备映射。
    
    返回值:
        model: 加载的多模态大语言模型
        processor: 对应的图像和文本处理器
    
    技术要点:
        - Qwen2.5-VL是阿里巴巴开源的视觉-语言多模态模型
        - device_map="auto"实现自动GPU分配，支持多卡推理
        - float32精度确保推理准确性（可选bfloat16以加速）
    """
    print("正在加载模型和处理器...")
    
    # 加载Qwen2.5-VL条件生成模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, 
        # torch_dtype=torch.bfloat16,  # 可选：使用bfloat16以节省显存
        torch_dtype=torch.float32,     # 使用float32确保精度
        device_map="auto"              # 自动分配GPU设备
    )
    
    # 加载对应的处理器（包含tokenizer和image processor）
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def perform_inference(data, model, processor):
    """
    对数据集执行推理
    
    参数:
        data: 输入数据列表，每个元素包含image_path和prompt
        model: 加载的多模态大语言模型
        processor: 对应的处理器
    
    返回值:
        results: 推理结果列表，包含原始数据和模型预测
    
    推理流程:
        1. 遍历数据集中的每个样本
        2. 构造多模态对话格式（图像+文本提示）
        3. 使用处理器对输入进行预处理
        4. 模型生成表情识别结果
        5. 解码并保存结果
    
    技术要点:
        - 采用对话式推理格式，模拟用户-助手交互
        - 图像统一缩放到224x224像素
        - do_sample=False确保确定性输出
        - max_new_tokens=1024限制生成长度
        - 使用torch.no_grad()节省显存
    """
    print("开始推理...")
    results = []
    
    # 使用tqdm显示进度条
    for item in tqdm(data, desc="处理图像中"):
        try:
            # ========== 准备输入 ==========
            image_path = item["image_path"]        # 图像路径
            original_prompt = item["prompt"]       # 原始提示词
            new_prompt = original_prompt
            
            # 构造多模态对话格式
            # 采用标准的对话格式：[{"role": "user", "content": [...]}]
            messages = [
                {
                    "role": "user",
                    "content": [
                        # 图像输入，统一缩放到224x224
                        {"type": "image", "image": image_path, "resized_height": 224, "resized_width": 224},
                        # 文本提示词
                        {"type": "text", "text": new_prompt}
                    ]
                }
            ]
            
            # ========== 处理输入 ==========
            # 应用对话模板，生成模型可接受的文本格式
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理视觉信息（加载和预处理图像）
            image_inputs, _ = process_vision_info(messages)
            
            # 使用处理器统一处理文本和图像
            inputs = processor(
                text=[text],
                images=image_inputs,
                return_tensors="pt"  # 返回PyTorch张量
            ).to(model.device)       # 移至模型所在设备
            
            # ========== 生成响应 ==========
            # 禁用梯度计算以节省显存
            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    do_sample=False,           # 使用贪婪解码，确保结果确定性
                    max_new_tokens=1024,       # 最多生成1024个token
                    use_cache=True             # 使用KV缓存加速生成
                )
            
            # ========== 处理输出 ==========
            # 移除输入部分，只保留生成的token
            generated_ids_trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
            
            # 解码生成的token为文本
            response = processor.decode(
                generated_ids_trimmed, 
                skip_special_tokens=True,          # 跳过特殊token
                clean_up_tokenization_spaces=False # 保持原始空格
            )
            
            # 保存结果（包含原始数据和模型响应）
            results.append({
                **item,                    # 原始数据（image_path, prompt等）
                "model_response": response # 模型生成的响应
            })
            
        except Exception as e:
            # 错误处理：记录错误信息
            print(f"处理 {image_path} 时出错: {str(e)}")
            results.append({
                **item,
                "model_response": f"ERROR: {str(e)}",
                "error": True
            })
    
    return results

if __name__ == "__main__":
    """
    主程序入口
    
    执行流程:
        1. 加载RAF-DB测试集数据
        2. 加载UniFER-7B模型
        3. 执行推理
        4. 保存结果到JSON文件
    """
    # ========== 1. 加载数据集 ==========
    print(f"正在加载数据集: {input_file}")
    with open(input_file, "r") as f:
        dataset = json.load(f)
    print(f"数据集加载完成，共 {len(dataset)} 个样本")
    
    # ========== 2. 加载模型 ==========
    model, processor = load_model()
    print("模型加载完成")
    
    # ========== 3. 执行推理 ==========
    results = perform_inference(dataset, model, processor)
    print(f"推理完成，共处理 {len(results)} 个样本")
    
    # ========== 4. 保存结果 ==========
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"推理结果已保存至: {output_file}")