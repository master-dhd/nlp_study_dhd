# -*- coding: utf-8 -*-

import torch
import logging
import json
from model_ner import create_ner_model
from config_ner import Config
from transformers import AutoTokenizer
from peft import get_peft_model, LoraConfig, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

"""
NER任务预测脚本
加载训练好的LoRA模型进行命名实体识别
"""

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NERPredictor:
    def __init__(self, config, model_path=None):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 标签映射
        self.id_to_label = config["id_to_label"]
        self.label_to_id = config["label_to_id"]
        
        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config["pretrain_model_path"])
        
        # 加载模型
        self.model = self.load_model(model_path)
        
        logger.info(f"NER预测器初始化完成，使用设备: {self.device}")
    
    def load_model(self, model_path=None):
        """
        加载训练好的模型
        """
        # 创建基础模型
        model = create_ner_model(self.config)
        
        # 应用PEFT配置
        tuning_tactics = self.config["tuning_tactics"]
        
        if tuning_tactics == "lora_tuning":
            peft_config = LoraConfig(
                r=self.config["lora_r"],
                lora_alpha=self.config["lora_alpha"],
                lora_dropout=self.config["lora_dropout"],
                target_modules=self.config["lora_target_modules"],
                task_type="TOKEN_CLS"
            )
        elif tuning_tactics == "p_tuning":
            peft_config = PromptEncoderConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "prompt_tuning":
            peft_config = PromptTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        elif tuning_tactics == "prefix_tuning":
            peft_config = PrefixTuningConfig(task_type="TOKEN_CLS", num_virtual_tokens=10)
        else:
            peft_config = None
        
        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        
        # 加载训练好的权重
        if model_path is None:
            model_path = f"output/best_{tuning_tactics}_ner.pth"
        
        try:
            logger.info(f"加载模型权重: {model_path}")
            loaded_weights = torch.load(model_path, map_location='cpu')
            
            # 更新模型权重
            model_state_dict = model.state_dict()
            model_state_dict.update(loaded_weights)
            model.load_state_dict(model_state_dict)
            
            logger.info("模型权重加载成功")
        except FileNotFoundError:
            logger.warning(f"模型文件不存在: {model_path}，使用未训练的模型")
        except Exception as e:
            logger.error(f"加载模型权重时出错: {e}")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict_text(self, text, return_confidence=False):
        """
        对单个文本进行NER预测
        """
        # 编码文本
        encoding = self.tokenizer(
            text,
            max_length=self.config["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).squeeze(0)
            
            if return_confidence:
                probabilities = torch.softmax(logits, dim=-1).squeeze(0)
                confidences = torch.max(probabilities, dim=-1)[0]
            else:
                confidences = None
        
        # 对齐预测结果与原文本
        entities = self.align_predictions_with_text(
            text, predictions.cpu(), offset_mapping, confidences.cpu() if confidences is not None else None
        )
        
        return entities
    
    def align_predictions_with_text(self, text, predictions, offset_mapping, confidences=None):
        """
        将预测结果与原文本对齐
        """
        entities = []
        current_entity = None
        
        for i, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 特殊token
                continue
            
            if i >= len(predictions):
                break
            
            pred_id = predictions[i].item()
            label = self.id_to_label.get(pred_id, 'O')
            confidence = confidences[i].item() if confidences is not None else None
            
            if label.startswith('B-'):
                # 开始新实体
                if current_entity is not None:
                    entities.append(current_entity)
                
                current_entity = {
                    'start': start.item(),
                    'end': end.item(),
                    'type': label[2:],
                    'text': text[start:end],
                    'confidence': confidence
                }
            elif label.startswith('I-') and current_entity is not None:
                # 继续当前实体
                if label[2:] == current_entity['type']:
                    current_entity['end'] = end.item()
                    current_entity['text'] = text[current_entity['start']:current_entity['end']]
                    if confidence is not None:
                        # 更新置信度（取平均值）
                        current_entity['confidence'] = (current_entity['confidence'] + confidence) / 2
                else:
                    # 实体类型不匹配，结束当前实体
                    entities.append(current_entity)
                    current_entity = None
            else:
                # O标签，结束当前实体
                if current_entity is not None:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        if current_entity is not None:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts, return_confidence=False):
        """
        批量预测
        """
        results = []
        for text in texts:
            entities = self.predict_text(text, return_confidence)
            results.append({
                'text': text,
                'entities': entities
            })
        return results
    
    def predict_file(self, input_file, output_file, return_confidence=False):
        """
        对文件中的文本进行预测
        """
        logger.info(f"开始处理文件: {input_file}")
        
        results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # 假设每行是一个JSON对象或纯文本
                    if line.startswith('{'):
                        data = json.loads(line)
                        text = data.get('text', '')
                    else:
                        text = line
                    
                    if text:
                        entities = self.predict_text(text, return_confidence)
                        results.append({
                            'line_num': line_num,
                            'text': text,
                            'entities': entities
                        })
                        
                        if line_num % 100 == 0:
                            logger.info(f"已处理 {line_num} 行")
                            
                except Exception as e:
                    logger.error(f"处理第 {line_num} 行时出错: {e}")
                    continue
        
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"预测完成，结果保存到: {output_file}")
        return results


def demo_prediction():
    """
    演示预测功能
    """
    logger.info("开始NER预测演示")
    
    # 创建预测器
    predictor = NERPredictor(Config)
    
    # 测试文本
    test_texts = [
        "张三在北京大学工作",
        "苹果公司的CEO蒂姆·库克访问了中国",
        "华为技术有限公司总部位于深圳市",
        "马云创办了阿里巴巴集团",
        "清华大学计算机系很有名"
    ]
    
    logger.info("开始预测测试文本...")
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n测试文本 {i}: {text}")
        
        entities = predictor.predict_text(text, return_confidence=True)
        
        if entities:
            logger.info("识别到的实体:")
            for entity in entities:
                logger.info(f"  - {entity['text']} ({entity['type']}) [{entity['start']}:{entity['end']}] 置信度: {entity['confidence']:.3f}")
        else:
            logger.info("  未识别到实体")
    
    logger.info("\n预测演示完成！")


if __name__ == "__main__":
    demo_prediction()
