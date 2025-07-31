# -*- coding: utf-8 -*-

import torch
import time
import json
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
from loader_qwen import load_qwen_data
import numpy as np
from collections import defaultdict

"""
Qwen模型评估器
用于评估标题预测文章内容的生成效果
"""

class QwenEvaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["bert_model_path"], 
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载验证数据
        self.eval_dataloader = load_qwen_data(
            config["eval_data_path"], 
            config, 
            logger, 
            shuffle=False
        )
        
        self.device = next(model.parameters()).device
    
    def eval(self, epoch):
        """
        评估模型性能
        """
        print(f"\n📊 开始第 {epoch} 轮评估...")
        eval_start_time = time.time()
        
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        generation_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.eval_dataloader):
                # 移动数据到设备
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item() * input_ids.size(0)
                total_samples += input_ids.size(0)
                
                # 收集生成示例（前3个批次）
                if batch_idx < 3:
                    self._collect_generation_examples(
                        input_ids, attention_mask, batch_idx, generation_examples
                    )
                
                # 显示进度
                if (batch_idx + 1) % 10 == 0:
                    current_loss = total_loss / total_samples
                    print(f"   批次 {batch_idx + 1}/{len(self.eval_dataloader)}, 当前平均损失: {current_loss:.4f}")
        
        # 计算平均损失
        avg_loss = total_loss / total_samples
        eval_time = time.time() - eval_start_time
        
        # 显示评估结果
        self._display_evaluation_results(epoch, avg_loss, eval_time, generation_examples)
        
        self.model.train()
        return avg_loss
    
    def _collect_generation_examples(self, input_ids, attention_mask, batch_idx, examples):
        """
        收集生成示例
        """
        try:
            # 取第一个样本
            sample_input = input_ids[0:1]
            sample_mask = attention_mask[0:1]
            
            # 找到"内容:"的位置，只保留标题部分作为输入
            input_text = self.tokenizer.decode(sample_input[0], skip_special_tokens=True)
            if "内容:" in input_text:
                title_part = input_text.split("内容:")[0] + "内容:"
                
                # 重新编码标题部分
                title_inputs = self.tokenizer(
                    title_part, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=50
                ).to(self.device)
                
                # 生成内容
                with torch.no_grad():
                    generated = self.model.generate(
                        **title_inputs,
                        max_new_tokens=80,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码生成的文本
                generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                original_text = self.tokenizer.decode(sample_input[0], skip_special_tokens=True)
                
                examples.append({
                    'batch_idx': batch_idx,
                    'original': original_text,
                    'generated': generated_text,
                    'title_part': title_part
                })
                
        except Exception as e:
            self.logger.warning(f"生成示例时出错: {e}")
    
    def _display_evaluation_results(self, epoch, avg_loss, eval_time, examples):
        """
        显示评估结果
        """
        print(f"\n📈 第 {epoch} 轮评估结果:")
        print(f"   平均损失: {avg_loss:.4f}")
        print(f"   评估耗时: {eval_time:.2f} 秒")
        print(f"   样本数量: {len(self.eval_dataloader.dataset)}")
        
        # 显示生成示例
        if examples:
            print(f"\n📝 生成示例:")
            for i, example in enumerate(examples[:3], 1):
                print(f"\n   示例 {i}:")
                
                # 提取标题
                if "标题:" in example['original']:
                    title = example['original'].split("标题:")[1].split("\n")[0].strip()
                    print(f"     标题: {title}")
                
                # 提取原始内容
                if "内容:" in example['original']:
                    original_content = example['original'].split("内容:")[1].strip()
                    print(f"     原始内容: {original_content[:100]}{'...' if len(original_content) > 100 else ''}")
                
                # 提取生成内容
                if "内容:" in example['generated']:
                    generated_content = example['generated'].split("内容:")[1].strip()
                    print(f"     生成内容: {generated_content[:100]}{'...' if len(generated_content) > 100 else ''}")
        
        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            print(f"\n💾 内存使用: {allocated:.2f} GB (缓存: {cached:.2f} GB)")
        
        print("─" * 50)
    
    def generate_sample(self, title, max_length=100):
        """
        根据标题生成内容示例
        """
        prompt = f"标题: {title}\n内容: "
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=50
        ).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取生成的内容部分
        if "内容:" in generated_text:
            content = generated_text.split("内容:")[1].strip()
            return content
        
        return generated_text
    
    def batch_evaluate_titles(self, titles):
        """
        批量评估标题生成效果
        """
        print(f"\n🔍 批量生成测试 ({len(titles)} 个标题)...")
        
        results = []
        total_time = 0
        
        for i, title in enumerate(titles, 1):
            print(f"\n📝 {i}/{len(titles)}: {title}")
            
            start_time = time.time()
            try:
                content = self.generate_sample(title)
                gen_time = time.time() - start_time
                total_time += gen_time
                
                print(f"   ⏱️  耗时: {gen_time:.2f}s")
                print(f"   📄 内容: {content[:150]}{'...' if len(content) > 150 else ''}")
                
                results.append({
                    'title': title,
                    'content': content,
                    'time': gen_time,
                    'success': True
                })
                
            except Exception as e:
                gen_time = time.time() - start_time
                total_time += gen_time
                print(f"   ❌ 生成失败: {str(e)}")
                
                results.append({
                    'title': title,
                    'content': '',
                    'time': gen_time,
                    'success': False
                })
        
        # 统计结果
        successful = sum(1 for r in results if r['success'])
        avg_time = total_time / len(results)
        
        print(f"\n📊 批量测试结果:")
        print(f"   成功率: {successful}/{len(titles)} ({successful/len(titles)*100:.1f}%)")
        print(f"   平均耗时: {avg_time:.2f} 秒")
        print(f"   总耗时: {total_time:.2f} 秒")
        
        return results


if __name__ == "__main__":
    from config_qwen import Config
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 这里可以添加独立的评估测试代码
    print("QwenEvaluator 模块已加载")
