# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import torch.nn as nn
import logging
from config_ner import Config
from model_ner import create_ner_model, choose_optimizer
from evaluate_ner import NEREvaluator
from loader_ner import load_ner_data
from peft import get_peft_model, LoraConfig, \
    PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
from transformers import get_linear_schedule_with_warmup

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
基于LoRA的NER任务训练主程序
"""

# 设置随机种子
seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载训练数据
    logger.info("加载训练数据...")
    train_data = load_ner_data(config, split='train')
    
    # 加载验证数据
    logger.info("加载验证数据...")
    valid_data = load_ner_data(config, split='validation', shuffle=False)
    
    # 创建模型
    logger.info("创建NER模型...")
    model = create_ner_model(config)
    
    # 应用LoRA微调策略
    tuning_tactics = config["tuning_tactics"]
    logger.info(f"应用微调策略: {tuning_tactics}")
    
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r=config["lora_r"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            task_type="TOKEN_CLS"  # NER任务类型
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(
            task_type="TOKEN_CLS", 
            num_virtual_tokens=10
        )
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(
            task_type="TOKEN_CLS", 
            num_virtual_tokens=10
        )
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            task_type="TOKEN_CLS", 
            num_virtual_tokens=10
        )
    else:
        peft_config = None
    
    if peft_config is not None:
        model = get_peft_model(model, peft_config)
        logger.info(f"LoRA配置应用成功，可训练参数数量: {model.num_parameters()}")
    
    # GPU设置
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可用，将模型迁移至GPU")
        model = model.cuda()
    
    # 设置优化器
    optimizer = choose_optimizer(config, model)
    
    # 设置学习率调度器
    total_steps = len(train_data) * config["epoch"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.get("warmup_steps", 0),
        num_training_steps=total_steps
    )
    
    # 创建评估器
    evaluator = NEREvaluator(config, model, logger)
    
    # 训练循环
    best_f1 = 0.0
    global_step = 0
    
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info(f"开始第 {epoch} 轮训练")
        
        train_loss = []
        
        for batch_idx, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            input_ids, attention_mask, labels = batch_data
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if config.get("max_grad_norm", 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            optimizer.step()
            scheduler.step()
            
            train_loss.append(loss.item())
            global_step += 1
            
            # 记录训练日志
            if global_step % config.get("logging_steps", 100) == 0:
                logger.info(f"Step {global_step}, Loss: {loss.item():.4f}")
            
            # 评估模型
            if global_step % config.get("eval_steps", 500) == 0:
                eval_results = evaluator.eval(epoch)
                f1_score = eval_results.get('f1', 0.0)
                
                # 保存最佳模型
                if f1_score > best_f1:
                    best_f1 = f1_score
                    model_path = os.path.join(config["model_path"], f"best_{tuning_tactics}_ner.pth")
                    save_tunable_parameters(model, model_path)
                    logger.info(f"保存最佳模型，F1: {best_f1:.4f}")
                
                model.train()  # 切换回训练模式
        
        # 每轮结束后的评估
        logger.info(f"第 {epoch} 轮平均损失: {np.mean(train_loss):.4f}")
        eval_results = evaluator.eval(epoch)
        
        # 保存检查点
        if epoch % config.get("save_epochs", 1) == 0:
            checkpoint_path = os.path.join(config["model_path"], f"checkpoint_epoch_{epoch}_{tuning_tactics}_ner.pth")
            save_tunable_parameters(model, checkpoint_path)
    
    # 保存最终模型
    final_model_path = os.path.join(config["model_path"], f"final_{tuning_tactics}_ner.pth")
    save_tunable_parameters(model, final_model_path)
    
    logger.info(f"训练完成！最佳F1分数: {best_f1:.4f}")
    return best_f1


def save_tunable_parameters(model, path):
    """
    保存可训练的参数
    """
    saved_params = {
        k: v.to("cpu")
        for k, v in model.named_parameters()
        if v.requires_grad
    }
    torch.save(saved_params, path)
    logger.info(f"模型参数已保存到: {path}")


if __name__ == "__main__":
    logger.info("开始基于LoRA的NER任务训练")
    logger.info(f"配置信息: {Config['task_type']}, 微调策略: {Config['tuning_tactics']}")
    
    try:
        best_score = main(Config)
        logger.info(f"训练成功完成，最佳分数: {best_score:.4f}")
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise
