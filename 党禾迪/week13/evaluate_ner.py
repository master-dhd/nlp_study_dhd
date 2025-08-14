# -*- coding: utf-8 -*-

import torch
import numpy as np
from collections import defaultdict
from loader_ner import load_ner_data
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

"""
NER任务评估器
支持实体级别和token级别的评估
"""


class NEREvaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        
        # 标签映射
        self.id_to_label = config["id_to_label"]
        self.label_to_id = config["label_to_id"]
        
        # 加载验证数据
        self.valid_data = load_ner_data(config, split='validation', shuffle=False)
        
        # 特殊标签ID
        self.pad_token_id = config.get("label_pad_token_id", -100)
        
        # 统计信息
        self.reset_stats()
    
    def reset_stats(self):
        """
        重置统计信息
        """
        self.stats = {
            "total_samples": 0,
            "total_tokens": 0,
            "correct_tokens": 0,
            "entities_true": [],
            "entities_pred": [],
            "labels_true": [],
            "labels_pred": []
        }
    
    def eval(self, epoch):
        """
        评估模型性能
        """
        self.logger.info(f"开始评估第 {epoch} 轮模型效果")
        
        self.model.eval()
        self.reset_stats()
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valid_data):
                if torch.cuda.is_available():
                    batch_data = [d.cuda() for d in batch_data]
                
                input_ids, attention_mask, labels = batch_data
                
                # 模型预测
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # 处理批次结果
                self.process_batch(
                    input_ids.cpu(),
                    attention_mask.cpu(),
                    labels.cpu(),
                    predictions.cpu()
                )
        
        # 计算评估指标
        results = self.compute_metrics()
        self.log_results(epoch, results)
        
        return results
    
    def process_batch(self, input_ids, attention_mask, true_labels, pred_labels):
        """
        处理一个批次的预测结果
        """
        batch_size = input_ids.size(0)
        
        for i in range(batch_size):
            # 获取有效的token（排除padding）
            valid_mask = attention_mask[i] == 1
            valid_true = true_labels[i][valid_mask]
            valid_pred = pred_labels[i][valid_mask]
            
            # 排除特殊标签
            mask = valid_true != self.pad_token_id
            valid_true = valid_true[mask]
            valid_pred = valid_pred[mask]
            
            if len(valid_true) == 0:
                continue
            
            # 转换为标签名称
            true_labels_str = [self.id_to_label.get(label.item(), 'O') for label in valid_true]
            pred_labels_str = [self.id_to_label.get(label.item(), 'O') for label in valid_pred]
            
            # 统计token级别准确率
            self.stats["total_tokens"] += len(valid_true)
            self.stats["correct_tokens"] += (valid_true == valid_pred).sum().item()
            
            # 收集标签用于分类报告
            self.stats["labels_true"].extend(true_labels_str)
            self.stats["labels_pred"].extend(pred_labels_str)
            
            # 提取实体用于实体级别评估
            true_entities = self.extract_entities(true_labels_str)
            pred_entities = self.extract_entities(pred_labels_str)
            
            self.stats["entities_true"].extend(true_entities)
            self.stats["entities_pred"].extend(pred_entities)
            
            self.stats["total_samples"] += 1
    
    def extract_entities(self, labels):
        """
        从BIO标签序列中提取实体
        """
        entities = []
        current_entity = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                # 开始新实体
                if current_entity is not None:
                    entities.append(current_entity)
                current_entity = {
                    'start': i,
                    'end': i + 1,
                    'type': label[2:]
                }
            elif label.startswith('I-') and current_entity is not None:
                # 继续当前实体
                if label[2:] == current_entity['type']:
                    current_entity['end'] = i + 1
                else:
                    # 实体类型不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            else:
                # O标签或其他，结束当前实体
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities
    
    def compute_metrics(self):
        """
        计算评估指标
        """
        results = {}
        
        # Token级别准确率
        if self.stats["total_tokens"] > 0:
            token_accuracy = self.stats["correct_tokens"] / self.stats["total_tokens"]
            results["token_accuracy"] = token_accuracy
        else:
            results["token_accuracy"] = 0.0
        
        # 标签级别的精确率、召回率、F1
        if len(self.stats["labels_true"]) > 0:
            # 获取所有标签（排除O）
            all_labels = list(set(self.stats["labels_true"] + self.stats["labels_pred"]))
            entity_labels = [label for label in all_labels if label != 'O']
            
            if entity_labels:
                # 计算宏平均
                precision_macro = precision_score(
                    self.stats["labels_true"], 
                    self.stats["labels_pred"], 
                    labels=entity_labels, 
                    average='macro', 
                    zero_division=0
                )
                recall_macro = recall_score(
                    self.stats["labels_true"], 
                    self.stats["labels_pred"], 
                    labels=entity_labels, 
                    average='macro', 
                    zero_division=0
                )
                f1_macro = f1_score(
                    self.stats["labels_true"], 
                    self.stats["labels_pred"], 
                    labels=entity_labels, 
                    average='macro', 
                    zero_division=0
                )
                
                # 计算微平均
                precision_micro = precision_score(
                    self.stats["labels_true"], 
                    self.stats["labels_pred"], 
                    labels=entity_labels, 
                    average='micro', 
                    zero_division=0
                )
                recall_micro = recall_score(
                    self.stats["labels_true"], 
                    self.stats["labels_pred"], 
                    labels=entity_labels, 
                    average='micro', 
                    zero_division=0
                )
                f1_micro = f1_score(
                    self.stats["labels_true"], 
                    self.stats["labels_pred"], 
                    labels=entity_labels, 
                    average='micro', 
                    zero_division=0
                )
                
                results.update({
                    "precision_macro": precision_macro,
                    "recall_macro": recall_macro,
                    "f1_macro": f1_macro,
                    "precision_micro": precision_micro,
                    "recall_micro": recall_micro,
                    "f1_micro": f1_micro,
                    "f1": f1_macro  # 主要指标
                })
        
        # 实体级别评估
        entity_results = self.compute_entity_metrics()
        results.update(entity_results)
        
        return results
    
    def compute_entity_metrics(self):
        """
        计算实体级别的评估指标
        """
        true_entities = set()
        pred_entities = set()
        
        # 将实体转换为可比较的格式
        for entity in self.stats["entities_true"]:
            true_entities.add((entity['start'], entity['end'], entity['type']))
        
        for entity in self.stats["entities_pred"]:
            pred_entities.add((entity['start'], entity['end'], entity['type']))
        
        # 计算精确率、召回率、F1
        if len(pred_entities) == 0:
            precision = 0.0
        else:
            precision = len(true_entities & pred_entities) / len(pred_entities)
        
        if len(true_entities) == 0:
            recall = 0.0
        else:
            recall = len(true_entities & pred_entities) / len(true_entities)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        return {
            "entity_precision": precision,
            "entity_recall": recall,
            "entity_f1": f1,
            "num_true_entities": len(true_entities),
            "num_pred_entities": len(pred_entities),
            "num_correct_entities": len(true_entities & pred_entities)
        }
    
    def log_results(self, epoch, results):
        """
        记录评估结果
        """
        self.logger.info(f"第 {epoch} 轮评估结果:")
        self.logger.info(f"  样本数量: {self.stats['total_samples']}")
        self.logger.info(f"  Token准确率: {results.get('token_accuracy', 0):.4f}")
        
        if 'f1_macro' in results:
            self.logger.info(f"  标签级F1 (宏平均): {results['f1_macro']:.4f}")
            self.logger.info(f"  标签级F1 (微平均): {results['f1_micro']:.4f}")
            self.logger.info(f"  标签级精确率: {results['precision_macro']:.4f}")
            self.logger.info(f"  标签级召回率: {results['recall_macro']:.4f}")
        
        if 'entity_f1' in results:
            self.logger.info(f"  实体级F1: {results['entity_f1']:.4f}")
            self.logger.info(f"  实体级精确率: {results['entity_precision']:.4f}")
            self.logger.info(f"  实体级召回率: {results['entity_recall']:.4f}")
            self.logger.info(f"  正确实体数: {results['num_correct_entities']} / {results['num_true_entities']}")
    
    def get_classification_report(self):
        """
        获取详细的分类报告
        """
        if len(self.stats["labels_true"]) == 0:
            return "无可用数据"
        
        # 获取实体标签
        all_labels = list(set(self.stats["labels_true"] + self.stats["labels_pred"]))
        entity_labels = [label for label in all_labels if label != 'O']
        
        if not entity_labels:
            return "无实体标签"
        
        report = classification_report(
            self.stats["labels_true"],
            self.stats["labels_pred"],
            labels=entity_labels,
            zero_division=0
        )
        
        return report


if __name__ == "__main__":
    # 测试评估器
    from config_ner import Config
    from model_ner import create_ner_model
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("测试NER评估器...")
    
    # 创建模型
    model = create_ner_model(Config)
    
    # 创建评估器
    evaluator = NEREvaluator(Config, model, logger)
    
    # 运行评估
    results = evaluator.eval(0)
    
    print("评估完成！")
    print(f"主要结果: {results}")
