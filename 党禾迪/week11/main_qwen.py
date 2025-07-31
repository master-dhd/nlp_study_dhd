# -*- coding: utf-8 -*-

import sys
import torch
import os
import random
import numpy as np
import time
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from config_qwen import Config
from evaluate_qwen import QwenEvaluator
from loader_qwen import load_qwen_data

# 设置PyTorch内存管理环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
Qwen2-7B模型训练主程序
用于标题预测文章内容的SFT训练
"""


def set_seed(seed):
    """
    设置随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_optimizer_and_scheduler(model, config, num_training_steps):
    """
    创建优化器和学习率调度器
    """
    # 不对bias和LayerNorm参数应用权重衰减
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config["learning_rate"],
        eps=config["adam_epsilon"]
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["warmup_steps"],
        num_training_steps=num_training_steps
    )
    
    return optimizer, scheduler


def train_epoch(model, dataloader, optimizer, scheduler, config, device, epoch, writer=None):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    num_steps = 0
    
    for step, batch in enumerate(dataloader):
        # 将数据移到设备上
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 前向传播
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        
        # 梯度累积
        if config["gradient_accumulation_steps"] > 1:
            loss = loss / config["gradient_accumulation_steps"]
        
        # 反向传播
        loss.backward()
        
        total_loss += loss.item()
        
        # 每个批次后清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # 监控GPU内存使用
            if step % 10 == 0:
                memory_allocated = torch.cuda.memory_allocated() / 1024**3
                memory_reserved = torch.cuda.memory_reserved() / 1024**3
                logger.info(f"Step {step}: GPU内存使用 {memory_allocated:.2f}GB / 预留 {memory_reserved:.2f}GB")
        
        # 梯度累积和优化
        if (step + 1) % config["gradient_accumulation_steps"] == 0:
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
            
            # 更新参数
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # 清理GPU缓存以防止内存累积
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            num_steps += 1
            
            # 记录日志
            if num_steps % config["logging_steps"] == 0:
                avg_loss = total_loss / num_steps
                current_lr = scheduler.get_last_lr()[0]
                
                try:
                    logger.info(f"Epoch {epoch}, Step {num_steps}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                except UnicodeEncodeError:
                    logger.info(f"Epoch {epoch}, Step {num_steps}, Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
                
                if writer:
                    global_step = (epoch - 1) * len(dataloader) + step
                    writer.add_scalar('Train/Loss', avg_loss, global_step)
                    writer.add_scalar('Train/LearningRate', current_lr, global_step)
    
    return total_loss / num_steps if num_steps > 0 else 0


def main(config):
    # 设置随机种子
    set_seed(config["seed"])
    
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 创建tensorboard日志目录
    log_dir = os.path.join(config["model_path"], "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)
    
    # 打印配置信息
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        logger.info(f"使用设备: {device}")
    except UnicodeEncodeError:
        logger.info(f"Using device: {device}")
    
    # 加载模型和tokenizer
    try:
        logger.info(f"加载{config['module_name']}模型: {config['bert_model_path']}")
    except UnicodeEncodeError:
        logger.info(f"Loading {config['module_name']} model: {config['bert_model_path']}")
    
    # 加载Qwen2-7B模型，使用内存优化设置
    model = AutoModelForCausalLM.from_pretrained(
        config["bert_model_path"],
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto",
        max_memory={0: "6GB"},  # 1.5B模型显存需求更低
        trust_remote_code=True,  # Qwen模型需要此参数
        use_cache=False  # 禁用缓存以节省内存
    )
    tokenizer = AutoTokenizer.from_pretrained(config["bert_model_path"], trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 模型已通过device_map自动分配到设备，无需手动移动
    
    # 启用梯度检查点以节省内存
    model.gradient_checkpointing_enable()
    
    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 加载训练数据
    try:
        logger.info("加载训练数据...")
    except UnicodeEncodeError:
        logger.info("Loading training data...")
    
    train_dataloader = load_qwen_data(config["train_data_path"], config, logger, shuffle=True)
    
    # 计算总训练步数
    num_training_steps = len(train_dataloader) * config["epoch"] // config["gradient_accumulation_steps"]
    
    # 创建优化器和调度器
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, num_training_steps)
    
    # 创建评估器
    evaluator = QwenEvaluator(config, model, logger)
    
    # 训练循环
    best_loss = float('inf')
    
    for epoch in range(1, config["epoch"] + 1):
        try:
            logger.info(f"开始第 {epoch} 轮训练")
        except UnicodeEncodeError:
            logger.info(f"Starting epoch {epoch}")
        
        # 训练一个epoch
        train_loss = train_epoch(
            model, train_dataloader, optimizer, scheduler, 
            config, device, epoch, writer
        )
        
        try:
            logger.info(f"第 {epoch} 轮训练完成，平均损失: {train_loss:.4f}")
        except UnicodeEncodeError:
            logger.info(f"Epoch {epoch} completed, average loss: {train_loss:.4f}")
        
        # 评估模型
        if epoch % config["eval_steps"] == 0 or epoch == config["epoch"]:
            eval_loss = evaluator.eval(epoch)
            
            if writer:
                writer.add_scalar('Eval/Loss', eval_loss, epoch)
            
            # 保存最佳模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_model_path = os.path.join(config["model_path"], "best_model")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)
                
                try:
                    logger.info(f"保存最佳模型到: {best_model_path}")
                except UnicodeEncodeError:
                    logger.info(f"Saved best model to: {best_model_path}")
        
        # 定期保存检查点
        if epoch % config["save_steps"] == 0:
            checkpoint_path = os.path.join(config["model_path"], f"checkpoint-epoch-{epoch}")
            model.save_pretrained(checkpoint_path)
            tokenizer.save_pretrained(checkpoint_path)
            
            try:
                logger.info(f"保存检查点到: {checkpoint_path}")
            except UnicodeEncodeError:
                logger.info(f"Saved checkpoint to: {checkpoint_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(config["model_path"], "final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    try:
        logger.info(f"训练完成！最终模型保存到: {final_model_path}")
    except UnicodeEncodeError:
        logger.info(f"Training completed! Final model saved to: {final_model_path}")
    
    writer.close()
    
    return model


if __name__ == "__main__":
    model = main(Config)
    
    # 测试生成功能
    try:
        logger.info("测试标题生成功能...")
    except UnicodeEncodeError:
        logger.info("Testing title generation...")
    
    evaluator = BertEvaluator(Config, model, logger)
    
    # 测试样例
    test_content = """阿根廷布宜诺斯艾利斯省奇尔梅斯市一服装店，8个月内被抢了三次。最后被抢劫的经历，更是直接让老板心理崩溃：歹徒在抢完不久后发现衣服"抢错了尺码"，理直气壮地拿着衣服到店里换，老板又不敢声张，只好忍气吞声。"""
    
    generated_title = evaluator.generate_title_beam_search(test_content)
    
    try:
        print(f"\n测试内容: {test_content[:50]}...")
        print(f"生成标题: {generated_title}")
    except UnicodeEncodeError:
        print(f"\nTest content: {test_content[:50].encode('utf-8', errors='ignore').decode('utf-8')}...")
        print(f"Generated title: {generated_title.encode('utf-8', errors='ignore').decode('utf-8')}")
