# -*- coding: utf-8 -*-

import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from collections import defaultdict

"""
BERT模型的数据加载器
使用MASK机制进行标题生成
"""


class BertDataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        
        # 初始化BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        
        # 特殊token的ID
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                line = json.loads(line)
                title = line["title"]
                content = line["content"]
                self.prepare_data(title, content)
        return

    def prepare_data(self, title, content):
        """
        准备BERT模型的训练数据
        格式: [CLS] content [SEP] title [SEP]
        对title部分进行MASK处理
        """
        # 对文本进行tokenize
        content_tokens = self.tokenizer.tokenize(content)
        title_tokens = self.tokenizer.tokenize(title)
        
        # 截断处理
        if len(content_tokens) > self.config["content_max_length"] - 2:  # 预留CLS和SEP
            content_tokens = content_tokens[:self.config["content_max_length"] - 2]
        
        if len(title_tokens) > self.config["title_max_length"] - 1:  # 预留SEP
            title_tokens = title_tokens[:self.config["title_max_length"] - 1]
        
        # 构建输入序列: [CLS] content [SEP] title [SEP]
        tokens = ["[CLS]"] + content_tokens + ["[SEP]"] + title_tokens + ["[SEP]"]
        
        # 转换为ID
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 创建attention mask
        attention_mask = [1] * len(input_ids)
        
        # 创建token type ids (0表示content部分，1表示title部分)
        token_type_ids = [0] * (len(content_tokens) + 2) + [1] * (len(title_tokens) + 1)
        
        # 创建标签，只对title部分计算loss
        labels = [-100] * (len(content_tokens) + 2) + input_ids[len(content_tokens) + 2:]
        
        # 对title部分进行MASK处理
        masked_input_ids = input_ids.copy()
        title_start_idx = len(content_tokens) + 2
        title_end_idx = len(input_ids) - 1  # 不包括最后的SEP
        
        for i in range(title_start_idx, title_end_idx):
            if random.random() < self.config["mask_prob"]:
                masked_input_ids[i] = self.mask_token_id
        
        # 填充到最大长度
        max_length = self.config["input_max_length"]
        
        # 填充
        padding_length = max_length - len(input_ids)
        if padding_length > 0:
            masked_input_ids.extend([self.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            token_type_ids.extend([0] * padding_length)
            labels.extend([-100] * padding_length)
        else:
            # 截断
            masked_input_ids = masked_input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            token_type_ids = token_type_ids[:max_length]
            labels = labels[:max_length]
        
        self.data.append({
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "original_title": title,
            "original_content": content
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def collate_fn(batch):
    """
    批处理函数
    """
    input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
    attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
    token_type_ids = torch.tensor([item["token_type_ids"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels
    }


def load_bert_data(data_path, config, logger, shuffle=True):
    """
    加载BERT数据
    """
    dataset = BertDataGenerator(data_path, config, logger)
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=0  # Windows上设置为0避免多进程问题
    )
    return dataloader


if __name__ == "__main__":
    from config_bert import Config
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    dl = load_bert_data(Config["train_data_path"], Config, logger)
    for batch in dl:
        print("Input IDs shape:", batch["input_ids"].shape)
        print("Attention Mask shape:", batch["attention_mask"].shape)
        print("Token Type IDs shape:", batch["token_type_ids"].shape)
        print("Labels shape:", batch["labels"].shape)
        break
