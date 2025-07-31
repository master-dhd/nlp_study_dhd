# -*- coding: utf-8 -*-

"""
BERT模型配置参数信息
"""
import os
import torch

module_name = 'Qwen/Qwen2-1.5B'  # 适合12GB显卡的1.5B模型，支持中英文，显存需求更低

Config = {
    "model_path": "output",
    "module_name": module_name,
    "bert_model_path": module_name,
    "input_max_length": 80,   # 输入+输出的总长度
    "content_max_length": 60,  # 文章内容最大长度
    "title_max_length": 20,   # 标题最大长度
    "epoch": 50,
    "batch_size": 2,  # 1.5B模型可以使用更大的批处理大小
    "optimizer": "adam",
    "learning_rate": 1.6e-4,  # gpt3-2.7B参数
    "warmup_steps": 500,
    "seed": 42,
    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",
    "eval_data_path": r"sample_data.json",
    "mask_prob": 0.15,  # MASK概率
    "save_steps": 100,
    "eval_steps": 100,
    "logging_steps": 50,
    "gradient_accumulation_steps": 4,  # 1.5B模型可以减少梯度累积步数
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
