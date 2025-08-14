# -*- coding: utf-8 -*-
"""
使用人民日报NER数据集的示例脚本
通过Hugging Face的datasets库加载标准的中文NER数据集
"""

import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from config import Config

def load_peoples_daily_ner():
    """
    加载人民日报NER数据集
    """
    print("正在加载人民日报NER数据集...")
    
    try:
        # 尝试多种加载方式
        dataset_names = [
            "peoples_daily_ner",
            "peoples-daily-ner/peoples_daily_ner", 
            "xusenlin/people-daily-ner"
        ]
        
        ner_datasets = None
        for dataset_name in dataset_names:
            try:
                print(f"尝试加载: {dataset_name}")
                ner_datasets = load_dataset(dataset_name, cache_dir="./data", trust_remote_code=True)
                print(f"成功加载数据集: {dataset_name}")
                break
            except Exception as e:
                print(f"加载 {dataset_name} 失败: {e}")
                continue
        
        if ner_datasets is None:
            print("所有数据集加载尝试都失败了")
            return None, None
        
        print("数据集加载成功！")
        print(f"可用分割: {list(ner_datasets.keys())}")
        
        # 检查数据集结构
        for split in ner_datasets.keys():
            print(f"{split}集样本数: {len(ner_datasets[split])}")
        
        # 查看数据格式
        print("\n数据格式示例:")
        first_split = list(ner_datasets.keys())[0]
        sample = ner_datasets[first_split][0]
        print(f"样本字段: {sample.keys()}")
        
        if 'tokens' in sample:
            print(f"tokens: {sample['tokens'][:10]}...")  # 显示前10个token
        if 'ner_tags' in sample:
            print(f"ner_tags: {sample['ner_tags'][:10]}...")  # 显示前10个标签
        if 'text' in sample:
            print(f"text: {sample['text'][:50]}...")  # 显示前50个字符
        if 'entities' in sample:
            print(f"entities: {sample['entities'][:3]}...")  # 显示前3个实体
        
        # 查看标签映射
        if 'ner_tags' in ner_datasets[first_split].features:
            label_names = ner_datasets[first_split].features['ner_tags'].feature.names
            print(f"\n标签类型: {label_names}")
        else:
            label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
            print(f"\n使用默认标签类型: {label_names}")
        
        return ner_datasets, label_names
        
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("请确保已安装datasets库: pip install datasets")
        print("如果问题持续，可能需要使用本地数据集")
        return None, None

def convert_to_our_format(ner_datasets, label_names, output_dir="./data"):
    """
    将人民日报NER数据集转换为我们项目使用的格式
    """
    if ner_datasets is None:
        return
    
    print("\n正在转换数据格式...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 转换各个数据集
    for split_name in ner_datasets.keys():
        dataset = ner_datasets[split_name]
        output_file = os.path.join(output_dir, f"peoples_daily_{split_name}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in dataset:
                if 'tokens' in sample and 'ner_tags' in sample:
                    # 标准格式
                    tokens = sample['tokens']
                    ner_tags = sample['ner_tags']
                    
                    # 将token列表转换为文本
                    text = ''.join(tokens)
                    
                    # 将数字标签转换为字符串标签
                    labels = [label_names[tag] if isinstance(tag, int) and tag < len(label_names) else str(tag) for tag in ner_tags]
                    
                elif 'text' in sample and 'entities' in sample:
                    # 实体格式
                    text = sample['text']
                    entities = sample['entities']
                    
                    # 将实体转换为BIO标签
                    labels = convert_entities_to_bio_labels(text, entities)
                    
                else:
                    print(f"未知的数据格式: {sample.keys()}")
                    continue
                
                # 保存为我们的格式
                json_line = {
                    "text": text,
                    "labels": labels
                }
                f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
        
        print(f"已保存 {split_name} 数据到: {output_file}")

def convert_entities_to_bio_labels(text, entities):
    """
    将实体列表转换为BIO标签序列
    """
    # 初始化所有字符为O标签
    labels = ['O'] * len(text)
    
    # 按实体位置排序，避免重叠问题
    # 支持不同的字段名格式
    def get_start_pos(entity):
        return entity.get('start', entity.get('start_offset', 0))
    
    sorted_entities = sorted(entities, key=get_start_pos)
    
    for entity in sorted_entities:
        # 支持不同的字段名格式
        start = entity.get('start', entity.get('start_offset'))
        end = entity.get('end', entity.get('end_offset'))
        entity_type = entity.get('label', entity.get('type', 'MISC'))
        
        # 确保索引在有效范围内
        if start is not None and end is not None and start >= 0 and end <= len(text) and start < end:
            # 设置B-标签
            labels[start] = f'B-{entity_type}'
            # 设置I-标签
            for i in range(start + 1, end):
                labels[i] = f'I-{entity_type}'
    
    return labels

def analyze_peoples_daily_data(ner_datasets, label_names):
    """
    分析人民日报NER数据集的统计信息
    """
    if ner_datasets is None:
        return
    
    print("\n=== 人民日报NER数据集分析 ===")
    
    for split_name in ner_datasets.keys():
        dataset = ner_datasets[split_name]
        print(f"\n{split_name.upper()} 集统计:")
        print(f"样本数量: {len(dataset)}")
        
        # 检查数据格式
        sample = dataset[0]
        print(f"数据字段: {list(sample.keys())}")
        
        if 'ner_tags' in sample:
            # 标准格式：tokens + ner_tags
            analyze_standard_format(dataset, label_names)
        elif 'entities' in sample:
            # 实体格式：text + entities
            analyze_entity_format(dataset)
        else:
            print("未知的数据格式")
            print(f"示例数据: {sample}")

def analyze_standard_format(dataset, label_names):
    """
    分析标准格式的数据（tokens + ner_tags）
    """
    label_counts = {}
    total_tokens = 0
    
    for sample in dataset:
        ner_tags = sample['ner_tags']
        total_tokens += len(ner_tags)
        
        for tag in ner_tags:
            if isinstance(tag, int):
                label_name = label_names[tag] if tag < len(label_names) else f"UNK_{tag}"
            else:
                label_name = tag
            label_counts[label_name] = label_counts.get(label_name, 0) + 1
    
    print(f"总token数: {total_tokens}")
    print(f"平均句长: {total_tokens/len(dataset):.1f}")
    
    print("标签分布:")
    for label, count in sorted(label_counts.items()):
        percentage = count / total_tokens * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

def analyze_entity_format(dataset):
    """
    分析实体格式的数据（text + entities）
    """
    entity_counts = {}
    total_chars = 0
    total_entities = 0
    
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        
        total_chars += len(text)
        total_entities += len(entities)
        
        for entity in entities:
            entity_type = entity.get('label', entity.get('type', 'UNKNOWN'))
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print(f"总字符数: {total_chars}")
    print(f"平均文本长度: {total_chars/len(dataset):.1f}")
    print(f"总实体数: {total_entities}")
    print(f"平均每句实体数: {total_entities/len(dataset):.1f}")
    
    print("实体类型分布:")
    for entity_type, count in sorted(entity_counts.items()):
        percentage = count / total_entities * 100 if total_entities > 0 else 0
        print(f"  {entity_type}: {count} ({percentage:.1f}%)")

def create_ner_loader_for_peoples_daily():
    """
    创建适用于人民日报NER数据集的数据加载器
    """
    loader_code = '''
# -*- coding: utf-8 -*-
"""
适用于人民日报NER数据集的数据加载器
"""

import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from datasets import load_dataset

class PeoplesDailyNERDataGenerator:
    def __init__(self, config, split='train'):
        self.config = config
        self.split = split
        
        # 加载人民日报NER数据集
        print(f"正在加载人民日报NER数据集 ({split})...")
        ner_datasets = load_dataset("peoples_daily_ner", cache_dir="./data")
        self.dataset = ner_datasets[split]
        
        # 获取标签映射
        self.label_names = self.dataset.features['ner_tags'].feature.names
        self.label_to_index = {label: idx for idx, label in enumerate(self.label_names)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
        
        # 更新配置
        self.config["class_num"] = len(self.label_names)
        
        # 初始化tokenizer
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        
        self.load()
    
    def load(self):
        self.data = []
        
        for sample in self.dataset:
            tokens = sample['tokens']
            ner_tags = sample['ner_tags']
            
            # 将token列表转换为文本
            text = ''.join(tokens)
            
            # 处理标签
            if self.config["model_type"] == "bert":
                # 使用BERT tokenizer进行编码
                encoded = self.tokenizer(
                    text,
                    max_length=self.config["max_length"],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = encoded['input_ids'].squeeze()
                
                # 对齐标签
                aligned_labels = self.align_labels_with_tokens(tokens, ner_tags, input_ids)
                
            else:
                # 使用字符级编码
                input_ids = self.encode_sentence(text)
                aligned_labels = ner_tags[:self.config["max_length"]]
                aligned_labels += [0] * (self.config["max_length"] - len(aligned_labels))
            
            input_ids = torch.LongTensor(input_ids)
            labels = torch.LongTensor(aligned_labels)
            
            self.data.append([input_ids, labels])
    
    def align_labels_with_tokens(self, original_tokens, original_labels, input_ids):
        """
        将原始标签与BERT tokenization后的token对齐
        """
        # 解码input_ids获取实际的tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        aligned_labels = []
        original_idx = 0
        
        for i, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                aligned_labels.append(0)  # O标签
            elif token.startswith('##'):
                # 子词，使用前一个标签但改为I-
                if aligned_labels and aligned_labels[-1] != 0:
                    prev_label = self.label_names[aligned_labels[-1]]
                    if prev_label.startswith('B-'):
                        new_label = 'I-' + prev_label[2:]
                        aligned_labels.append(self.label_to_index.get(new_label, 0))
                    else:
                        aligned_labels.append(aligned_labels[-1])
                else:
                    aligned_labels.append(0)
            else:
                # 正常token
                if original_idx < len(original_labels):
                    aligned_labels.append(original_labels[original_idx])
                    original_idx += 1
                else:
                    aligned_labels.append(0)
        
        # 确保长度匹配
        aligned_labels = aligned_labels[:self.config["max_length"]]
        aligned_labels += [0] * (self.config["max_length"] - len(aligned_labels))
        
        return aligned_labels
    
    def encode_sentence(self, text):
        # 字符级编码（如果不使用BERT）
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id
    
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

def load_peoples_daily_data(config, split='train', shuffle=True):
    """
    加载人民日报NER数据
    """
    dg = PeoplesDailyNERDataGenerator(config, split)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
'''
    
    with open('loader_peoples_daily.py', 'w', encoding='utf-8') as f:
        f.write(loader_code)
    
    print("已创建人民日报NER数据加载器: loader_peoples_daily.py")

if __name__ == "__main__":
    print("=== 人民日报NER数据集使用示例 ===")
    
    # 1. 加载数据集
    ner_datasets, label_names = load_peoples_daily_ner()
    
    if ner_datasets is not None:
        # 2. 分析数据集
        analyze_peoples_daily_data(ner_datasets, label_names)
        
        # 3. 转换数据格式
        convert_to_our_format(ner_datasets, label_names)
        
        # 4. 创建数据加载器
        create_ner_loader_for_peoples_daily()
        
        print("\n=== 使用建议 ===")
        print("1. 安装依赖: pip install datasets")
        print("2. 运行此脚本加载和转换数据")
        print("3. 修改config.py中的数据路径")
        print("4. 使用loader_peoples_daily.py替换原有的loader.py")
        print("5. 调整model.py使用AutoModelForTokenClassification")
    else:
        print("\n请先安装datasets库: pip install datasets")
