# -*- coding: utf-8 -*-            
# @Time : 2025/6/25 22:04
# @Author : CodeDi
# @FileName: bert参数量计算.py

import torch
from transformers import BertModel


def count_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_params_per_layer(model):
    """返回各子模块参数量的 dict。"""
    counts = {"embeddings": sum(p.numel() for p in model.embeddings.parameters())}
    counts.update({f"layer_{i + 1}": sum(p.numel() for p in l.parameters())
                   for i, l in enumerate(model.encoder.layer)})
    counts["pooler"] = sum(p.numel() for p in model.pooler.parameters())
    return counts

def human_bytes(num_bytes, unit="MB"):
    """把字节换成 MB 或 GB（默认 MB）。"""
    div = 1024 ** 2 if unit.upper() == "MB" else 1024 ** 3
    return num_bytes / div

if __name__ == "__main__":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = BertModel.from_pretrained("bert-base-uncased").to(device).eval()
        layer_cnt = count_params_per_layer(model)
        total_params = sum(layer_cnt.values())

        print("\n各层参数量：")
        for k, v in layer_cnt.items():
            print(f"{k:<12}: {v:,}")

        # === 内存估算 ===
        bytes_fp32 = total_params * 4  # 4 B/param
        bytes_fp16 = total_params * 2  # 2 B/param

        print(f"\n模型总参数量 : {total_params:,}")
        print(f"≈ {human_bytes(bytes_fp32):.2f} MB (float32)")
        print(f"≈ {human_bytes(bytes_fp16):.2f} MB (float16)")

        print("====="*20)

        # 加载预训练的bert-base-uncased模型
        model = BertModel.from_pretrained('bert-base-uncased')

        # 计算参数量
        total, trainable = count_parameters(model)

        print(f"模型总参数量：{total:,} 个")
        print(f"可训练参数量：{trainable:,} 个")
