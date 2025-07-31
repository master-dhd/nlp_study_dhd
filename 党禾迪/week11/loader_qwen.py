# -*- coding: utf-8 -*-

import torch
import json
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class QwenDataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["bert_model_path"], trust_remote_code=True)
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # 加载数据
        self.data = self.load_data()
        
    def load_data(self):
        """
        加载训练数据
        """
        data = []
        try:
            with open(self.path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        item = json.loads(line)
                        if 'title' in item and 'content' in item:
                            data.append(item)
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return []
        
        self.logger.info(f"加载了 {len(data)} 条数据")
        return data
    
    def prepare_data(self, title, content):
        """
        准备Qwen模型的训练数据
        格式: 标题：{title}\n内容：{content}
        """
        # 构建prompt
        prompt = f"标题：{title}\n内容："
        full_text = prompt + content
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.config["input_max_length"],
            padding=False,
            return_tensors=None
        )
        
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        
        # 创建labels，只对标题部分计算loss
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.config["input_max_length"],
            padding=False,
            return_tensors=None
        )
        
        prompt_length = len(prompt_encoding['input_ids'])
        
        # labels: prompt部分设为-100，标题部分保持原值
        labels = [-100] * prompt_length + input_ids[prompt_length:]
        
        # 确保长度一致
        if len(labels) != len(input_ids):
            labels = labels[:len(input_ids)]
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return self.prepare_data(item['title'], item['content'])

class QwenDataset(Dataset):
    def __init__(self, data_generator):
        self.data_generator = data_generator
    
    def __len__(self):
        return len(self.data_generator)
    
    def __getitem__(self, idx):
        return self.data_generator[idx]

def collate_fn(batch, pad_token_id):
    """
    批处理函数
    """
    # 获取最大长度
    max_length = max(len(item['input_ids']) for item in batch)
    
    # 填充
    input_ids = []
    attention_mask = []
    labels = []
    
    for item in batch:
        # 填充input_ids
        padded_input_ids = item['input_ids'] + [pad_token_id] * (max_length - len(item['input_ids']))
        input_ids.append(padded_input_ids)
        
        # 填充attention_mask
        padded_attention_mask = item['attention_mask'] + [0] * (max_length - len(item['attention_mask']))
        attention_mask.append(padded_attention_mask)
        
        # 填充labels
        padded_labels = item['labels'] + [-100] * (max_length - len(item['labels']))
        labels.append(padded_labels)
    
    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'labels': torch.tensor(labels, dtype=torch.long)
    }

def load_qwen_data(data_path, config, logger, shuffle=True):
    """
    加载Qwen模型的数据
    """
    data_generator = QwenDataGenerator(data_path, config, logger)
    dataset = QwenDataset(data_generator)
    
    # 创建collate_fn
    def collate_wrapper(batch):
        return collate_fn(batch, data_generator.pad_token_id)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        collate_fn=collate_wrapper,
        num_workers=0  # Windows上设为0避免多进程问题
    )
    
    return dataloader


if __name__ == "__main__":
    from config_qwen import Config
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 测试数据加载
    try:
        dl = load_qwen_data(Config["train_data_path"], Config, logger)
        logger.info(f"数据加载器创建成功，数据集大小: {len(dl.dataset)}")
        
        for i, batch in enumerate(dl):
            logger.info(f"批次 {i+1}:")
            logger.info(f"  Input IDs shape: {batch['input_ids'].shape}")
            logger.info(f"  Attention Mask shape: {batch['attention_mask'].shape}")
            logger.info(f"  Labels shape: {batch['labels'].shape}")
            
            # 显示第一个样本的内容
            if i == 0:
                tokenizer = AutoTokenizer.from_pretrained(Config["bert_model_path"], trust_remote_code=True)
                sample_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
                logger.info(f"  样本文本: {sample_text}")
            
            if i >= 2:  # 只显示前3个批次
                break
                
    except Exception as e:
        logger.error(f"测试失败: {e}")
