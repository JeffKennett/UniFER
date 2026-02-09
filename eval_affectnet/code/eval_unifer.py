"""
UniFER AffectNet 数据集评估脚本
=================================

功能说明:
    本脚本用于评估UniFER模型在AffectNet数据集上的表情识别性能。
    从推理结果中提取预测标签，计算各项性能指标。

主要功能:
    1. 从模型响应中提取表情标签
    2. 计算整体准确率、精确率、召回率、F1分数
    3. 生成混淆矩阵
    4. 计算每个表情类别的详细指标
    5. 保存评估结果

评估指标:
    - 准确率 (Accuracy): 正确预测的样本比例
    - 精确率 (Precision): 预测为正类中真正为正类的比例
    - 召回率 (Recall): 真正为正类中被预测为正类的比例  
    - F1分数 (F1-Score): 精确率和召回率的调和平均值
    - 混淆矩阵 (Confusion Matrix): 展示预测与真实标签的对应关系

作者: UniFER团队
日期: 2025
"""

import json
import re
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ==================== 配置参数 ====================
# 输入文件: 推理结果JSON文件
input_file = "./UniFER/eval_affectnet/results/affectnet_unifer_7b_results.json"

# 输出文件: 保存评估指标
output_file = "./UniFER/eval_affectnet/results/affectnet_unifer_7b_metrics.json"

def extract_label(response, candidate_labels):
    """
    从模型响应中提取表情标签
    
    参数:
        response: 模型生成的文本响应
        candidate_labels: 候选表情标签列表
    
    返回值:
        提取出的表情标签，如果无法识别则返回"unknown"
    
    提取策略（按优先级顺序）:
        1. 优先从<answer></answer>标签中提取答案
        2. 在响应文本中搜索完整单词匹配
        3. 搜索部分匹配
        4. 使用同义词映射进行模糊匹配
        
    技术要点:
        - 使用正则表达式解析结构化响应
        - 大小写不敏感匹配
        - 支持表情同义词映射（如happy -> happiness）
        - 边界词匹配确保准确性
    """
    # ========== 策略1: 从<answer>标签中提取 ==========
    # 使用正则表达式匹配<answer>标签内的内容
    # re.IGNORECASE: 忽略大小写
    # re.DOTALL: 让.匹配包括换行符在内的所有字符
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', response, re.IGNORECASE)
    
    if answer_match:
        # 提取并清理答案文本
        answer_text = answer_match.group(1).strip().lower()
     
        # 精确匹配候选标签
        for label in candidate_labels:
            if label.lower() == answer_text.lower():
                return label

            # 部分匹配（标签包含在答案中）
            if label.lower() in answer_text.lower():
                return label
    
    # ========== 策略2: 在整个响应中搜索标签 ==========
    response_lower = response.lower()
    for label in candidate_labels:
        # 完整单词匹配（使用\b边界符）
        # 例如: \bhappiness\b 匹配 "happiness" 但不匹配 "unhappiness"
        if re.search(rf'\b{label}\b', response_lower):
            return label
        
        # 部分匹配（备选方案）
        if label in response_lower:
            return label
    
    # ========== 策略3: 使用同义词映射 ==========
    # 定义每种表情的同义词列表
    # 用于处理模型可能使用的不同表达方式
    mapping = {
        "anger": ["angry"],                                    # 愤怒
        "disgust": ["revulsion", "repulsion"],                # 厌恶
        "fear": ["terror", "fright"],                         # 恐惧
        "happiness": ["joy", "happy"],                        # 快乐
        "sadness": ["sad"],                                   # 悲伤
        "neutral": ["normal", "calm", "blank"],               # 中性
        "surprise": ["shock", "astonishment"]                 # 惊讶
    }
    
    # 遍历映射表，查找同义词
    for label, alternatives in mapping.items():
        if any(alt in response_lower for alt in alternatives):
            return label
    
    # ========== 无法识别 ==========
    # 如果所有策略都失败，返回"unknown"
    return "unknown"

def calculate_metrics(results):
    """
    计算并展示评估指标
    
    参数:
        results: 推理结果列表，包含true_label和model_response
    
    功能:
        1. 从模型响应中提取预测标签
        2. 计算整体性能指标（准确率、精确率、召回率、F1）
        3. 生成混淆矩阵
        4. 计算每个类别的详细指标
        5. 按F1分数排序展示类别性能
        6. 保存结果到JSON文件
    
    输出:
        - 终端打印详细的评估报告
        - JSON文件保存数值结果
    """
    true_labels = []      # 真实标签列表
    pred_labels = []      # 预测标签列表
    errors = 0            # 错误样本计数
    unknowns = 0          # 无法识别的预测计数
    
    # ========== 遍历结果，提取标签 ==========
    for item in results:
        # 跳过推理失败的样本
        if "error" in item and item["error"]:
            errors += 1
            continue
            
        true_label = item["true_label"]              # 真实标签
        model_response = item["model_response"]      # 模型响应
        candidate_labels = item["candidate_labels"]  # 候选标签列表
        
        # 从响应中提取预测标签
        pred_label = extract_label(model_response, candidate_labels)
        
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
        # 统计无法识别的预测
        if pred_label == "unknown":
            unknowns += 1
    
    # ========== 打印基本统计信息 ==========
    print(f"\n总样本数: {len(results)}")
    print(f"错误样本: {errors}")
    print(f"无法识别的预测: {unknowns}")
    print(f"有效预测: {len(true_labels)}")
    
    # ========== 计算整体性能指标 ==========
    # 准确率: 正确预测的比例
    accuracy = accuracy_score(true_labels, pred_labels)
    
    # 宏平均指标: 每个类别指标的平均值（不考虑类别不平衡）
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, pred_labels, 
        average='macro',           # 宏平均
        zero_division=0            # 避免除零错误
    )
    
    # ========== 生成混淆矩阵 ==========
    # 混淆矩阵: 行表示真实标签，列表示预测标签
    # 对角线元素表示正确预测的数量
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=candidate_labels)
    
    # ========== 生成详细分类报告 ==========
    # 包含每个类别的precision, recall, f1-score, support
    class_report = classification_report(
        true_labels, 
        pred_labels, 
        labels=candidate_labels,
        zero_division=0,
        target_names=candidate_labels,
        output_dict=True  # 返回字典格式
    )
    
    # ========== 打印整体性能指标 ==========
    print("\n===== 整体性能指标 =====")
    print(f"准确率 (Accuracy): {accuracy:.4f}")
    print(f"精确率 (Precision, macro): {precision:.4f}")
    print(f"召回率 (Recall, macro): {recall:.4f}")
    print(f"F1分数 (F1-Score, macro): {f1:.4f}")
    
    # ========== 打印混淆矩阵 ==========
    print("\n===== 混淆矩阵 =====")
    print("真实标签 (行) vs 预测标签 (列)")
    
    # 打印列标题
    print(f"{'':>10}", end="")
    for label in candidate_labels:
        print(f"{label[:5]:>8}", end="")  # 只显示前5个字符
    print("\n" + "-" * (10 + len(candidate_labels)*8))
    
    # 打印矩阵内容
    for i, true_label in enumerate(candidate_labels):
        print(f"{true_label[:10]:>10}", end="")  # 行标签
        for j, pred_label in enumerate(candidate_labels):
            print(f"{conf_matrix[i][j]:>8}", end="")  # 矩阵元素
        print()
    
    # ========== 打印每个类别的性能 ==========
    print("\n===== 各类别性能指标 =====")
    print(f"{'类别':<12} {'精确率':<10} {'召回率':<10} {'F1分数':<10} {'样本数':<10}")
    
    # 收集每个类别的统计信息
    per_class_stats = []
    
    for emotion in candidate_labels:
        stats = class_report[emotion]
        precision_c = stats['precision']  # 该类别的精确率
        recall_c = stats['recall']        # 该类别的召回率
        f1_c = stats['f1-score']          # 该类别的F1分数
        support_c = stats['support']      # 该类别的样本数
        
        per_class_stats.append({
            "emotion": emotion,
            "precision": precision_c,
            "recall": recall_c,
            "f1": f1_c,
            "support": support_c
        })
        
        print(f"{emotion:<12} {precision_c:<10.4f} {recall_c:<10.4f} {f1_c:<10.4f} {support_c:<10.0f}")
    
    # ========== 按F1分数排序展示 ==========
    sorted_stats = sorted(per_class_stats, key=lambda x: x['f1'], reverse=True)
    
    print("\n===== 类别按F1分数排序 =====")
    print(f"{'排名':<5} {'类别':<12} {'F1分数':<10}")
    for rank, stats in enumerate(sorted_stats, 1):
        print(f"{rank:<5} {stats['emotion']:<12} {stats['f1']:<10.4f}")
    
    # ========== 保存结果到JSON ==========
    detailed_results = {
        "overall": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "per_class": per_class_stats,
        "confusion_matrix": conf_matrix.tolist()
    }
    
    # 写入文件
    with open(output_file, "w") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    print(f"\n评估指标已保存至: {output_file}")

if __name__ == "__main__":
    """
    主程序入口
    
    执行流程:
        1. 加载推理结果JSON文件
        2. 计算并展示评估指标
    """
    print(f"正在加载推理结果: {input_file}")
    with open(input_file, "r") as f:
        results = json.load(f)
    print(f"推理结果加载完成，共 {len(results)} 个样本")
    
    # 计算评估指标
    calculate_metrics(results)