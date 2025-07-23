# -*- coding: utf-8 -*-

"""
BERT模型配置参数信息
"""
import os
import torch

Config = {
    "model_path": "output",
    "bert_model_path": r"C:\Users\Administrator\.cache\modelscope\hub\models\google-bert\bert-base-chinese",
    "input_max_length": 150,  # 输入+输出的总长度
    "content_max_length": 120,  # 文章内容最大长度
    "title_max_length": 30,   # 标题最大长度
    "epoch": 50,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT通常使用较小的学习率
    "warmup_steps": 500,
    "seed": 42,
    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",
    "mask_prob": 0.15,  # MASK概率
    "save_steps": 100,
    "eval_steps": 100,
    "logging_steps": 50,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "adam_epsilon": 1e-8,
    "num_beams": 5,  # beam search的beam数量
    "early_stopping": True,
    "do_sample": False,
    "temperature": 1.0,
    "top_k": 50,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "length_penalty": 1.0,
    "no_repeat_ngram_size": 2
}
