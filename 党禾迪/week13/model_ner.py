# -*- coding: utf-8 -*-
"""
NER任务模型定义
使用AutoModelForTokenClassification进行序列标注
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoConfig
)
from torch.optim import Adam, SGD, AdamW
from config_ner import Config

def create_ner_model(config):
    """
    创建NER模型
    """
    # 加载预训练模型配置
    model_config = AutoConfig.from_pretrained(
        config["pretrain_model_path"],
        num_labels=config["num_labels"],
        id2label=config["id_to_label"],
        label2id=config["label_to_id"]
    )
    
    # 创建TokenClassification模型
    model = AutoModelForTokenClassification.from_pretrained(
        config["pretrain_model_path"],
        config=model_config
    )
    
    return model

# 创建全局模型实例
TorchModel = create_ner_model(Config)

class NERModel(nn.Module):
    """
    自定义NER模型包装器
    """
    def __init__(self, config):
        super(NERModel, self).__init__()
        self.config = config
        self.num_labels = config["num_labels"]
        
        # 加载预训练BERT模型
        self.bert = AutoModelForTokenClassification.from_pretrained(
            config["pretrain_model_path"],
            num_labels=self.num_labels
        )
        
        # Dropout层
        self.dropout = nn.Dropout(config.get("hidden_dropout_prob", 0.1))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        return outputs

def choose_optimizer(config, model):
    """
    选择优化器
    """
    optimizer_name = config["optimizer"].lower()
    learning_rate = config["learning_rate"]
    weight_decay = config.get("weight_decay", 0.0)
    
    # 获取需要训练的参数
    if hasattr(model, 'named_parameters'):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.parameters()
    
    if optimizer_name == "adam":
        return Adam(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=0.9)
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

def compute_ner_loss(predictions, labels, attention_mask=None, label_pad_token_id=-100):
    """
    计算NER损失
    """
    # 如果模型输出包含损失，直接返回
    if hasattr(predictions, 'loss') and predictions.loss is not None:
        return predictions.loss
    
    # 否则手动计算损失
    if hasattr(predictions, 'logits'):
        logits = predictions.logits
    else:
        logits = predictions
    
    # 展平logits和labels
    active_loss = attention_mask.view(-1) == 1 if attention_mask is not None else None
    active_logits = logits.view(-1, logits.shape[-1])
    active_labels = labels.view(-1)
    
    if active_loss is not None:
        active_logits = active_logits[active_loss]
        active_labels = active_labels[active_loss]
    
    # 过滤掉padding的标签
    if label_pad_token_id is not None:
        active_labels = active_labels[active_labels != label_pad_token_id]
        active_logits = active_logits[labels.view(-1) != label_pad_token_id]
    
    loss_fct = nn.CrossEntropyLoss()
    loss = loss_fct(active_logits, active_labels)
    
    return loss

def get_tokenizer(config):
    """
    获取tokenizer
    """
    return AutoTokenizer.from_pretrained(config["pretrain_model_path"])

class NERTokenizer:
    """
    NER任务的tokenizer包装器
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
        self.label_to_id = config["label_to_id"]
        self.id_to_label = config["id_to_label"]
        
    def encode_plus_aligned(
        self, 
        text, 
        labels=None, 
        max_length=None, 
        padding="max_length", 
        truncation=True,
        return_tensors="pt"
    ):
        """
        编码文本并对齐标签
        """
        if max_length is None:
            max_length = self.config["max_length"]
            
        # 编码文本
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors=return_tensors,
            return_offsets_mapping=True if labels is not None else False
        )
        
        if labels is not None:
            # 对齐标签
            aligned_labels = self.align_labels_with_tokens(
                text, labels, encoded['offset_mapping'][0]
            )
            encoded['labels'] = torch.tensor(aligned_labels, dtype=torch.long)
            
            # 移除offset_mapping（不需要传给模型）
            del encoded['offset_mapping']
        
        return encoded
    
    def align_labels_with_tokens(self, text, labels, offset_mapping):
        """
        将字符级标签与token对齐
        """
        aligned_labels = []
        label_idx = 0
        
        for start, end in offset_mapping:
            if start == 0 and end == 0:  # [CLS], [SEP], [PAD]
                aligned_labels.append(self.config.get("label_pad_token_id", -100))
            else:
                # 找到对应的字符标签
                if label_idx < len(labels):
                    aligned_labels.append(self.label_to_id.get(labels[label_idx], 0))
                    if end - start == 1:  # 单个字符
                        label_idx += 1
                else:
                    aligned_labels.append(self.config.get("label_pad_token_id", -100))
        
        return aligned_labels
    
    def decode_labels(self, label_ids):
        """
        将标签ID解码为标签名称
        """
        return [self.id_to_label.get(label_id, "O") for label_id in label_ids]

if __name__ == "__main__":
    # 测试模型创建
    print("测试NER模型创建...")
    
    model = create_ner_model(Config)
    print(f"模型类型: {type(model)}")
    print(f"标签数量: {model.num_labels}")
    
    # 测试tokenizer
    tokenizer = NERTokenizer(Config)
    test_text = "张三在北京大学工作"
    test_labels = ["B-PER", "I-PER", "O", "B-ORG", "I-ORG", "I-ORG", "O", "O"]
    
    encoded = tokenizer.encode_plus_aligned(test_text, test_labels)
    print(f"\n测试编码结果:")
    print(f"input_ids shape: {encoded['input_ids'].shape}")
    print(f"labels shape: {encoded['labels'].shape}")
    
    print("\nNER模型测试完成！")
