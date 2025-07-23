# -*- coding: utf-8 -*-

import torch
import json
from transformers import BertTokenizer, BertForMaskedLM
from loader_bert import load_bert_data
from collections import defaultdict

"""
BERT模型效果测试
"""


class BertEvaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        
        # 初始化tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_model_path"])
        
        # 加载验证数据
        self.valid_data = load_bert_data(config["valid_data_path"], config, logger, shuffle=False)
        
        # 特殊token
        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.mask_token_id = self.tokenizer.mask_token_id

    def eval(self, epoch):
        """
        评估模型效果
        """
        try:
            self.logger.info("开始测试第%d轮模型效果：" % epoch)
        except UnicodeEncodeError:
            self.logger.info("Starting evaluation for epoch %d:" % epoch)
            
        self.model.eval()
        self.model.cpu()
        
        total_loss = 0
        num_samples = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.valid_data):
                if batch_idx >= 5:  # 只评估前5个batch
                    break
                    
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                token_type_ids = batch["token_type_ids"]
                labels = batch["labels"]
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                num_samples += 1
                
                # 生成标题示例
                if batch_idx == 0:
                    self.generate_title_examples(input_ids[0:1], attention_mask[0:1], token_type_ids[0:1])
        
        avg_loss = total_loss / num_samples if num_samples > 0 else 0
        try:
            self.logger.info(f"验证集平均损失: {avg_loss:.4f}")
        except UnicodeEncodeError:
            self.logger.info(f"Validation average loss: {avg_loss:.4f}")
        
        return avg_loss

    def generate_title_examples(self, input_ids, attention_mask, token_type_ids):
        """
        生成标题示例
        """
        try:
            # 找到SEP token的位置，确定content和title的边界
            sep_positions = (input_ids[0] == self.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) < 2:
                return
            
            content_end = sep_positions[0].item()
            title_start = content_end + 1
            title_end = sep_positions[1].item()
            
            # 提取原始内容
            content_tokens = input_ids[0][1:content_end]  # 跳过CLS
            original_content = self.tokenizer.decode(content_tokens, skip_special_tokens=True)
            
            # 创建用于生成的输入（将title部分全部MASK）
            masked_input = input_ids.clone()
            for i in range(title_start, title_end):
                masked_input[0][i] = self.mask_token_id
            
            # 逐步生成标题
            generated_title = self.generate_title_iteratively(
                masked_input, attention_mask, token_type_ids, title_start, title_end
            )
            
            try:
                print(f"原文内容: {original_content[:100]}...")
                print(f"生成标题: {generated_title}")
                print("-" * 50)
            except UnicodeEncodeError:
                print(f"Content: {original_content[:100].encode('utf-8', errors='ignore').decode('utf-8')}...")
                print(f"Generated Title: {generated_title.encode('utf-8', errors='ignore').decode('utf-8')}")
                print("-" * 50)
                
        except Exception as e:
            self.logger.error(f"生成标题示例时出错: {e}")

    def generate_title_iteratively(self, input_ids, attention_mask, token_type_ids, title_start, title_end):
        """
        迭代生成标题
        """
        self.model.eval()
        
        with torch.no_grad():
            current_input = input_ids.clone()
            
            # 迭代预测每个MASK位置
            for step in range(10):  # 最多迭代10次
                outputs = self.model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                predictions = outputs.logits
                
                # 找到第一个MASK token的位置
                mask_positions = (current_input[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    break
                
                # 预测第一个MASK位置的token
                mask_pos = mask_positions[0].item()
                if title_start <= mask_pos < title_end:
                    predicted_token_id = predictions[0, mask_pos].argmax(dim=-1).item()
                    current_input[0, mask_pos] = predicted_token_id
        
        # 提取生成的标题
        title_tokens = current_input[0][title_start:title_end]
        generated_title = self.tokenizer.decode(title_tokens, skip_special_tokens=True)
        
        return generated_title

    def generate_title_beam_search(self, content_text, num_beams=5):
        """
        使用beam search生成标题
        """
        # 对内容进行tokenize
        content_tokens = self.tokenizer.tokenize(content_text)
        if len(content_tokens) > self.config["content_max_length"] - 2:
            content_tokens = content_tokens[:self.config["content_max_length"] - 2]
        
        # 构建输入序列
        max_title_length = self.config["title_max_length"]
        tokens = ["[CLS]"] + content_tokens + ["[SEP]"] + ["[MASK]"] * max_title_length + ["[SEP]"]
        
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        attention_mask = torch.ones_like(input_ids)
        
        # 创建token type ids
        content_length = len(content_tokens) + 2
        token_type_ids = torch.cat([
            torch.zeros(1, content_length),
            torch.ones(1, len(tokens) - content_length)
        ], dim=1).long()
        
        # 生成标题
        title_start = content_length
        title_end = len(tokens) - 1
        
        generated_title = self.generate_title_iteratively(
            input_ids, attention_mask, token_type_ids, title_start, title_end
        )
        
        return generated_title


if __name__ == "__main__":
    from config_bert import Config
    from transformers import BertForMaskedLM
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 加载模型
    model = BertForMaskedLM.from_pretrained(Config["bert_model_path"])
    
    # 创建评估器
    evaluator = BertEvaluator(Config, model, logger)
    
    # 测试生成
    test_content = "阿根廷布宜诺斯艾利斯省奇尔梅斯市一服装店，8个月内被抢了三次。"
    generated_title = evaluator.generate_title_beam_search(test_content)
    print(f"生成的标题: {generated_title}")
