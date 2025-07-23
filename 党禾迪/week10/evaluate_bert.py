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

    def _is_valid_token(self, token_id):
        """
        检查token是否为有效字符
        """
        # 排除特殊token
        if token_id in [self.pad_token_id, self.mask_token_id, self.cls_token_id, self.sep_token_id]:
            return False
        
        # 获取token对应的文本
        try:
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
            if not token_text:
                return False
            
            # 排除明确的无效符号
            invalid_chars = {'#', '♀', '♂', '★', '☆', '※', '○', '●', '◎', '◇', '◆', '□', '■', '△', '▲', '▽', '▼', 
                           '§', '¶', '†', '‡', '•', '‰', '′', '″', '‴', '※', '‼', '⁇', '⁈', '⁉'}
            if any(char in invalid_chars for char in token_text):
                return False
            
            # 首先排除包含日文字符的token
            for char in token_text:
                if ('\u3040' <= char <= '\u309f' or  # 平假名
                    '\u30a0' <= char <= '\u30ff'):   # 片假名
                    return False
            
            # 排除一些常见的繁体字符，优先使用简体中文
            traditional_chars = {'荘', '覇', '敗', '莊', '優', '勝', '庄', '勝', '優', '敗', '莊', '荘'}
            if any(char in traditional_chars for char in token_text):
                return False
            
            # 检查是否包含有效字符（主要是简体中文）
            has_valid_char = False
            for char in token_text:
                # 优先选择常用简体中文字符
                if '\u4e00' <= char <= '\u9fff':  # 中文汉字范围
                    # 进一步过滤，优先选择常用简体字
                    if char in '的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞':
                        has_valid_char = True
                        break
                # 允许数字和基本标点
                elif char.isdigit() or char in '，。！？：；、（）【】《》':
                    has_valid_char = True
                    break
            
            return has_valid_char
        except:
            return False

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
            used_tokens = set()  # 记录已使用的token，避免重复
            
            # 迭代预测每个MASK位置，直到没有MASK或达到最大迭代次数
            max_iterations = title_end - title_start  # 最多迭代title长度次
            for step in range(max_iterations):
                outputs = self.model(
                    input_ids=current_input,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                predictions = outputs.logits
                
                # 找到所有MASK token的位置
                mask_positions = (current_input[0] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if len(mask_positions) == 0:
                    break
                
                # 预测所有MASK位置的token，选择概率最高且未重复的位置进行替换
                best_score = -float('inf')
                best_pos = None
                best_token = None
                
                for mask_pos in mask_positions:
                    mask_pos = mask_pos.item()
                    if title_start <= mask_pos < title_end:
                        # 获取该位置的预测概率分布
                        probs = torch.softmax(predictions[0, mask_pos], dim=-1)
                        
                        # 获取top-k候选token，避免总是选择最高概率的
                        top_k = min(10, probs.size(0))
                        top_probs, top_indices = torch.topk(probs, top_k)
                        
                        for i, (prob, token_id) in enumerate(zip(top_probs, top_indices)):
                            token_id = token_id.item()
                            
                            # 避免重复使用相同token（除了常见字符）
                            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True).strip()
                            if token_text in used_tokens and len(used_tokens) > 2:
                                continue
                            
                            # 检查token有效性
                            if self._is_valid_token(token_id):
                                # 给予多样性奖励，降低重复token的分数
                                diversity_bonus = 0.1 if token_text not in used_tokens else -0.2
                                adjusted_score = prob.item() + diversity_bonus
                                
                                if adjusted_score > best_score:
                                    best_score = adjusted_score
                                    best_pos = mask_pos
                                    best_token = token_id
                            else:
                                # 降级策略
                                if token_id not in [self.pad_token_id, self.mask_token_id, self.cls_token_id, self.sep_token_id]:
                                    has_japanese = any(('\u3040' <= char <= '\u309f' or '\u30a0' <= char <= '\u30ff') for char in token_text)
                                    has_special = any(char in '##♀♂★☆※○●◎◇◆□■△▲▽▼' for char in token_text)
                                    # 检查繁体字符
                                    traditional_chars = {'荘', '覇', '敗', '莊', '優', '勝', '庄', '勝', '優', '敗', '莊', '荘'}
                                    has_traditional = any(char in traditional_chars for char in token_text)
                                    if token_text and not has_japanese and not has_special and not has_traditional:
                                        if token_text not in used_tokens:
                                            adjusted_score = prob.item() - 0.1  # 降级分数
                                            if adjusted_score > best_score:
                                                best_score = adjusted_score
                                                best_pos = mask_pos
                                                best_token = token_id
                
                # 替换最佳位置的token
                if best_pos is not None and best_token is not None:
                    current_input[0, best_pos] = best_token
                    token_text = self.tokenizer.decode([best_token], skip_special_tokens=True).strip()
                    if token_text:
                        used_tokens.add(token_text)
                else:
                    break
        
        # 提取生成的标题
        title_tokens = current_input[0][title_start:title_end]
        generated_title = self.tokenizer.decode(title_tokens, skip_special_tokens=True)
        
        return generated_title

    def generate_title_beam_search(self, content_text, num_beams=5):
        """
        使用改进的生成策略生成标题
        """
        # 对内容进行tokenize
        content_tokens = self.tokenizer.tokenize(content_text)
        if len(content_tokens) > self.config["content_max_length"] - 2:
            content_tokens = content_tokens[:self.config["content_max_length"] - 2]
        
        # 构建输入序列，使用较短的title长度
        max_title_length = min(15, self.config["title_max_length"])  # 限制标题长度
        tokens = ["[CLS]"] + content_tokens + ["[SEP]"] + ["[MASK]"] * max_title_length + ["[SEP]"]
        
        input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokens)])
        attention_mask = torch.ones_like(input_ids)
        
        # 创建token type ids
        content_length = len(content_tokens) + 2
        token_type_ids = torch.cat([
            torch.zeros(1, content_length),
            torch.ones(1, len(tokens) - content_length)
        ], dim=1).long()
        
        # 移动到设备
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        
        # 生成标题
        title_start = content_length
        title_end = len(tokens) - 1
        
        generated_title = self.generate_title_iteratively(
            input_ids, attention_mask, token_type_ids, title_start, title_end
        )
        
        # 清理生成的标题
        generated_title = generated_title.strip()
        if not generated_title or len(generated_title) < 2:
            return "无法生成标题"
        
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
