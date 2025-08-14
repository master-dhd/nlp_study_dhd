# -*- coding: utf-8 -*-

"""
NER任务配置参数信息 - 适用于人民日报NER数据集
"""

Config = {
    "model_path": "output",
    
    # 使用人民日报NER数据集
    "use_peoples_daily": True,  # 是否使用人民日报数据集
    "train_data_path": "data/peoples_daily_train.json",  # 转换后的训练数据
    "valid_data_path": "data/peoples_daily_validation.json",  # 转换后的验证数据
    "test_data_path": "data/peoples_daily_test.json",  # 转换后的测试数据
    
    # 或者使用我们自己创建的数据
    "custom_train_data_path": "data/train_ner_data.json",
    "custom_valid_data_path": "data/valid_ner_data.json",
    
    "vocab_path": "chars.txt",
    "model_type": "bert",
    
    # NER任务特定参数
    "task_type": "ner",  # 任务类型：ner
    "max_length": 128,  # NER任务通常需要更长的序列
    "label_pad_token_id": -100,  # 用于padding的标签ID
    
    # 模型参数
    "hidden_size": 768,  # BERT-base的hidden size
    "num_layers": 12,
    
    # 训练参数
    "epoch": 5,  # NER任务通常不需要太多epoch
    "batch_size": 16,  # 适合NER任务的batch size
    "learning_rate": 2e-5,  # 适合BERT微调的学习率
    
    # 微调策略
    "tuning_tactics": "lora_tuning",
    # "tuning_tactics": "full_finetuning",
    
    # LoRA参数
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["query", "key", "value", "dense"],
    
    # 优化器
    "optimizer": "adamw",  # AdamW更适合Transformer模型
    "weight_decay": 0.01,
    "warmup_steps": 500,
    
    # 模型路径
    "pretrain_model_path": "bert-base-chinese",  # 使用Hugging Face模型名称
    
    # 评估参数
    "eval_steps": 500,  # 每500步评估一次
    "save_steps": 1000,  # 每1000步保存一次
    "logging_steps": 100,  # 每100步记录一次
    
    # 其他参数
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    
    # NER标签（人民日报数据集的标签）
    "ner_labels": [
        "O",
        "B-PER", "I-PER",  # 人名
        "B-LOC", "I-LOC",  # 地名
        "B-ORG", "I-ORG",  # 组织机构
    ],
    
    # 自定义标签（如果使用我们创建的数据集）
    "custom_ner_labels": [
        "O",
        "B-PER", "I-PER",     # 人名
        "B-LOC", "I-LOC",     # 地名
        "B-ORG", "I-ORG",     # 组织机构
        "B-TIME", "I-TIME",   # 时间
        "B-MONEY", "I-MONEY", # 金额
        "B-PROD", "I-PROD",   # 产品
        "B-PERCENT", "I-PERCENT", # 百分比
    ]
}

# 根据使用的数据集设置标签
if Config["use_peoples_daily"]:
    Config["labels"] = Config["ner_labels"]
    Config["train_data_path"] = Config["train_data_path"]
    Config["valid_data_path"] = Config["valid_data_path"]
else:
    Config["labels"] = Config["custom_ner_labels"]
    Config["train_data_path"] = Config["custom_train_data_path"]
    Config["valid_data_path"] = Config["custom_valid_data_path"]

# 设置类别数量
Config["class_num"] = len(Config["labels"])
Config["num_labels"] = Config["class_num"]

# 创建标签到索引的映射
Config["label_to_id"] = {label: idx for idx, label in enumerate(Config["labels"])}
Config["id_to_label"] = {idx: label for label, idx in Config["label_to_id"].items()}

print(f"配置加载完成 - 任务类型: {Config['task_type']}")
print(f"使用数据集: {'人民日报NER' if Config['use_peoples_daily'] else '自定义NER'}")
print(f"标签数量: {Config['class_num']}")
print(f"微调策略: {Config['tuning_tactics']}")
