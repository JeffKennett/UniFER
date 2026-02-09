# UniFER 项目文档中心

欢迎来到UniFER项目的中文文档中心！本目录包含了项目的所有中文文档，包括技术细节、使用指南、面试准备等。

---

## 📚 文档列表

### 1. [项目流程文档](./项目流程文档.md) ⭐
**最重要的文档**，详细介绍了整个项目的完整流程：
- 项目背景和创新点
- 完整的项目结构
- 数据准备流程
- 两阶段训练详解（SFT + GRPO）
- 模型评估流程
- 技术细节和原理

**适合**: 首次了解项目、理解整体架构、复现实验

---

### 2. [使用文档](./使用文档.md) 🚀
从零开始的完整使用指南：
- 环境配置和依赖安装
- 数据准备步骤
- 模型训练详细步骤
- 推理评估操作指南
- 进阶使用技巧
- 性能优化建议
- 故障排查和常见问题

**适合**: 实际操作、复现实验、部署应用

---

### 3. [面试知识点总结](./面试知识点总结.md) 🎓
针对技术面试的知识点整理：
- 深度学习基础（优化器、正则化、Transformer等）
- 多模态模型原理
- 大语言模型技术
- 强化学习（GRPO）
- 分布式训练（DeepSpeed）
- 评估指标详解
- 常见面试问题及回答

**适合**: 准备技术面试、巩固理论知识

---

### 4. [简历项目描述](./简历项目描述.md) 📝
如何在简历中展示这个项目：
- 多种简历描述模板
- 技术亮点总结
- 项目成果量化
- 面试常见问题及回答
- 项目展示建议
- 简历优化建议
- 相关证明材料

**适合**: 撰写简历、准备面试、项目展示

---

## 🎯 快速导航

### 我想...

#### 了解项目
→ 阅读 [项目流程文档](./项目流程文档.md) 第一部分

#### 复现实验
→ 按照 [使用文档](./使用文档.md) 的步骤操作

#### 准备面试
→ 学习 [面试知识点总结](./面试知识点总结.md)  
→ 参考 [简历项目描述](./简历项目描述.md) 的面试问题部分

#### 写简历
→ 使用 [简历项目描述](./简历项目描述.md) 中的模板

#### 深入理解技术
→ [项目流程文档](./项目流程文档.md) 的技术细节部分  
→ [面试知识点总结](./面试知识点总结.md) 的理论知识

#### 优化性能
→ [使用文档](./使用文档.md) 的性能优化部分

#### 解决问题
→ [使用文档](./使用文档.md) 的故障排查部分

---

## 📊 文档关系图

```
简历项目描述 ──→ 面试知识点总结
     ↓                  ↓
项目流程文档 ←──── 使用文档
```

**建议阅读顺序**:
1. **项目流程文档** - 理解全貌
2. **使用文档** - 动手实践
3. **面试知识点总结** - 理论深化
4. **简历项目描述** - 应用展示

---

## 🔍 代码注释说明

项目中的所有Python代码都已添加详细的中文注释，包括：

### 评估脚本
- `eval_rafdb/code/infer_unifer.py` - RAF-DB推理
- `eval_rafdb/code/eval_unifer.py` - RAF-DB评估
- `eval_ferplus/code/` - FERPlus评估
- `eval_affectnet/code/` - AffectNet评估
- `eval_sfew_2.0/code/` - SFEW 2.0评估
- `eval_total/code/eval_unifer.py` - 总体评估

### 训练脚本
- `train_unifer/src/r1-v/src/open_r1/sft_fer.py` - SFT训练
- `train_unifer/src/r1-v/src/open_r1/grpo.py` - GRPO训练
- `train_unifer/src/r1-v/src/open_r1/trainer/` - 自定义训练器

每个文件都包含：
- 文件级文档字符串（功能说明、技术栈等）
- 函数级文档字符串（参数、返回值、流程等）
- 行内注释（技术要点、注意事项等）

---

## 💡 重要概念速查

### 核心技术
- **MLLM**: 多模态大语言模型
- **CoT**: 思维链推理 (Chain-of-Thought)
- **SFT**: 监督微调 (Supervised Fine-Tuning)
- **GRPO**: 群体相对策略优化 (Group Relative Policy Optimization)
- **RLVR**: 可验证奖励强化学习

### 关键数据集
- **FERBench**: 评估基准（4个数据集）
- **UniFER-CoT-230K**: 思维链数据集
- **UniFER-RLVR-360K**: 强化学习数据集

### 性能指标
- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1-Score**: F1分数
- **Confusion Matrix**: 混淆矩阵

---

## 🔗 相关资源

### 项目资源
- **GitHub**: https://github.com/zfkarl/UniFER
- **论文**: https://arxiv.org/abs/2511.00389
- **模型**: https://huggingface.co/Karl28/UniFER-7B
- **数据集**: https://huggingface.co/datasets/Karl28/UniFER

### 技术依赖
- **Qwen2.5-VL**: https://github.com/QwenLM/Qwen2-VL
- **R1-V**: https://github.com/StarsfieldAI/R1-V
- **DeepSpeed**: https://www.deepspeed.ai/
- **vLLM**: https://github.com/vllm-project/vllm
- **TRL**: https://github.com/huggingface/trl

---

## 📧 联系方式

- **邮箱**: fzhang@link.cuhk.edu.hk
- **GitHub Issues**: https://github.com/zfkarl/UniFER/issues

---

## 📜 许可证

本文档遵循项目的开源许可证。

---

## 🎉 贡献

欢迎提出文档改进建议！可以通过以下方式：
1. 提交GitHub Issue
2. 发送邮件反馈
3. 提交Pull Request

---

**最后更新**: 2025年1月

**文档版本**: v1.0

**祝学习愉快！如有问题，随时联系。**
