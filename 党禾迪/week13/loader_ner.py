# -*- coding: utf-8 -*-

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from config_ner import Config

"""
NER任务数据加载器
支持人民日报NER数据集和自定义NER数据集
"""


class NERDataGenerator(Dataset):
    def __init__(self, data_path, config, split='train'):
        self.config = config
        self.data_path = data_path
        self.split = split
        
        # 标签映射
        self.label_to_id = config["label_to_id"]
        self.id_to_label = config["id_to_label"]
        self.num_labels = config["num_labels"]
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
        
        # 特殊标签ID
        self.pad_token_id = config.get("label_pad_token_id", -100)
        
        # 加载数据
        self.load_data()
        
        print(f"加载 {split} 数据完成，样本数量: {len(self.data)}")
    
    def load_data(self):
        """
        加载NER数据
        """
        self.data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    line = line.strip()
                    if not line:
                        continue
                    
                    sample = json.loads(line)
                    text = sample['text']
                    labels = sample['labels']
                    
                    # 编码文本和标签
                    encoded = self.encode_text_and_labels(text, labels)
                    if encoded is not None:
                        self.data.append(encoded)
                        
                except Exception as e:
                    print(f"处理第 {line_num+1} 行数据时出错: {e}")
                    continue
    
    def encode_text_and_labels(self, text, labels):
        """
        编码文本和标签
        """
        # 检查文本和标签长度是否匹配
        if len(text) != len(labels):
            print(f"文本长度 ({len(text)}) 与标签长度 ({len(labels)}) 不匹配: {text[:50]}...")
            return None
        
        # 使用tokenizer编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        
        # 对齐标签
        aligned_labels = self.align_labels_with_tokens(text, labels, offset_mapping)
        aligned_labels = torch.tensor(aligned_labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': aligned_labels
        }
    
    def align_labels_with_tokens(self, text, labels, offset_mapping):
        """
        将字符级标签与token对齐
        """
        aligned_labels = []
        
        for start, end in offset_mapping:
            if start == 0 and end == 0:  # 特殊token [CLS], [SEP], [PAD]
                aligned_labels.append(self.pad_token_id)
            else:
                # 获取对应的字符标签
                if start < len(labels):
                    label = labels[start]
                    label_id = self.label_to_id.get(label, self.label_to_id.get('O', 0))
                    aligned_labels.append(label_id)
                else:
                    aligned_labels.append(self.pad_token_id)
        
        return aligned_labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return item['input_ids'], item['attention_mask'], item['labels']


def load_ner_data(config, split='train', shuffle=True):
    """
    加载NER数据
    """
    # 根据split确定数据路径
    if split == 'train':
        data_path = config["train_data_path"]
    elif split == 'validation' or split == 'valid':
        data_path = config["valid_data_path"]
    elif split == 'test':
        data_path = config.get("test_data_path", config["valid_data_path"])
    else:
        raise ValueError(f"不支持的数据分割: {split}")
    
    # 创建数据生成器
    dataset = NERDataGenerator(data_path, config, split)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=0,  # Windows下设置为0
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader


class NERCollator:
    """
    NER数据的批处理整理器
    """
    def __init__(self, pad_token_id=-100):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        input_ids = [item[0] for item in batch]
        attention_masks = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        
        # 转换为tensor
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        labels = torch.stack(labels)
        
        return input_ids, attention_masks, labels


def test_ner_loader():
    """
    测试NER数据加载器
    """
    print("测试NER数据加载器...")
    
    # 加载训练数据
    train_loader = load_ner_data(Config, split='train')
    print(f"训练数据批次数: {len(train_loader)}")
    
    # 测试一个批次
    for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx}:")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  attention_mask shape: {attention_mask.shape}")
        print(f"  labels shape: {labels.shape}")
        
        # 显示第一个样本的详细信息
        if batch_idx == 0:
            print(f"  第一个样本的input_ids: {input_ids[0][:20]}...")
            print(f"  第一个样本的labels: {labels[0][:20]}...")
        
        if batch_idx >= 2:  # 只测试前3个批次
            break
    
    print("NER数据加载器测试完成！")


if __name__ == "__main__":
    test_ner_loader()
