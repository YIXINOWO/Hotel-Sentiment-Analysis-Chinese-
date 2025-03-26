# Hotel-Sentiment-Analysis-Chinese-
# 中文情感分析项目 (Chinese Sentiment Analysis)  基于BERT的中文酒店评论情感分析模型，使用Transformers和PyTorch实现。  ## 项目简介  本项目实现了一个二分类的情感分析模型，可以自动判断中文酒店评论的情感倾向（正面/负面）。模型基于预训练的中文BERT模型，通过微调实现了较高的分类准确率。  ## 性能指标  - 训练集准确率: ~100% - 测试集准确率: ~90% - 训练损失: 从0.7降至接近0  ## 环境要求
bash
Python >= 3.6
torch
transformers
pandas
numpy
matplotlib
seaborn
scikit-learn

## 数据集

使用ChnSentiCorp酒店评论数据集：
- 包含正面和负面的中文酒店评论
- 标签：0（负面）和1（正面）
- 数据集大小：约12000条评论

## 模型架构

- 基础模型：bert-base-chinese
- 分类头：线性层
- 激活函数：Softmax
- 损失函数：CrossEntropyLoss

## 训练参数

- 批次大小：32
- 学习率：2e-5
- 训练轮数：5
- 优化器：AdamW
- 最大序列长度：128

## 使用方法

1. 数据预处理：
python
from src.data_processor import prepare_data
train_loader, test_loader = prepare_data('data/hotel_reviews.csv')

2. 训练模型：
python
from src.model import SentimentClassifier
from src.train import train_model
model = SentimentClassifier()
trained_model = train_model(model, train_loader, test_loader, device)

3. 预测示例：
python
text = "这家酒店的服务很好，环境优美"
result = predict_sentiment(text, model, device)
print(f"情感倾向: {result['sentiment']}")
print(f"置信度: {result['probability']:.2%}")


## 可视化结果

项目包含两个主要的可视化图表：
1. 模型准确率随时间变化图
2. 训练损失随时间变化图

这些图表显示了模型的训练过程和性能提升情况。

## 性能分析

1. 优点：
- 收敛速度快
- 测试集准确率高
- 训练过程稳定

2. 局限性：
- 存在轻微过拟合
- 仅支持二分类
- 对长文本处理能力有限

## 未来改进

1. 技术改进：
- 添加更多正则化方法
- 实现学习率调度
- 增加验证集
- 尝试其他预训练模型

2. 功能扩展：
- 支持多分类
- 添加情感强度分析
- 实现批量处理
  
