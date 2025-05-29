# -*- coding: utf-8 -*-
# @Time : 2025/5/29 19:08
# @Author : CodiX
# @FileName: rnn分类.py

# 导入必要的库
import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pad_sequence  # 用于对变长序列进行填充

# 构建词表：将字符映射为编号
def build_vocab():
    """
    构造一个简单的字符到编号的词表。
    - 包括 'a'~'l' 字符
    - 添加 'pad' 和 'unk' 表示填充和未知字符
    """
    chars = "abcdefghjkl"  # 字符集
    vocab = {"pad": 0}  # padding 符号
    for i, c in enumerate(chars):
        vocab[c] = i + 1  # 每个字符对应一个唯一编号
    vocab['unk'] = len(vocab)  # 未知字符的编号
    return vocab

# 构造单个样本：生成随机字符串，并标记 a 首次出现的位置
def build_sample(vocab, length=6):
    """
    随机生成一个长度为 length 的字符串，并返回其对应的索引列表和标签 y。

    标签 y：
    - 如果 'a' 出现在字符串中，则为它第一次出现的索引位置 (0~5)
    - 否则为 5，表示没有 'a'
    """
    x = [random.choice(list(vocab.keys())) for _ in range(length)]  # 随机选择字符
    first_a_index = -1
    for idx, char in enumerate(x):  # 找出第一个 'a'
        if char == 'a':
            first_a_index = idx
            break
    y = first_a_index if first_a_index != -1 else 5  # 设置标签
    x_ids = [vocab.get(c, vocab['unk']) for c in x]  # 将字符转换为编号
    return x_ids, y

# 构造数据集
def build_dataset(num_samples=10000):
    """
    构造训练数据集。
    返回：
    - dataset_x: 输入张量列表
    - dataset_y: 对应的标签列表
    - vocab: 使用的词表
    """
    dataset_x = []
    dataset_y = []
    vocab = build_vocab()
    for _ in range(num_samples):
        x, y = build_sample(vocab)
        dataset_x.append(torch.LongTensor(x))
        dataset_y.append(y)
    dataset_x = pad_sequence(dataset_x, batch_first=True)  # 填充成相同长度
    return dataset_x, torch.LongTensor(dataset_y), vocab

# 定义 RNN 分类模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_classes):
        super(RNNClassifier, self).__init__()
        # 嵌入层：将字符编号映射为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # RNN 层：处理序列数据
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        # 分类器：输出每个样本属于哪一类
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        前向传播函数
        参数：
        - x: 输入张量 shape = (batch_size, seq_len)

        返回：
        - logits: 输出 logit，shape = (batch_size, num_classes)
        """
        x = self.embedding(x)  # shape = (batch_size, seq_len, embed_dim)
        out, hidden = self.rnn(x)  # out shape = (batch_size, seq_len, hidden_size)
        logits = self.classifier(out[:, 0, :])  # 只使用第一个时间步的隐藏状态做分类
        return logits

# 训练函数
def train():
    """
    主训练函数，完成以下任务：
    - 数据加载与划分
    - 模型初始化
    - 模型训练
    - 模型保存
    """
    # 超参数设置
    embed_dim = 5       # 每个字用 5 维向量表示
    hidden_size = 32    # RNN 隐藏层大小
    num_classes = 6     # 分成 6 类（0~5 是位置，5 表示无 a）
    num_samples = 10000 # 训练样本数量
    epochs = 50         # 训练轮数
    batch_size = 64     # 批次大小
    learning_rate = 0.0001  # 学习率
    val_split = 0.2     # 验证集比例

    # 构造数据集
    x, y, vocab = build_dataset(num_samples)
    vocab_size = len(vocab)  # 正确获取词表大小
    dataset = torch.utils.data.TensorDataset(x, y)

    # 划分训练集和验证集
    val_size = int(val_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = RNNClassifier(vocab_size, embed_dim, hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()  # 分类任务常用交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        # 验证阶段
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = model(x_batch)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == y_batch).sum().item()
                total_val += y_batch.size(0)

        print(f"Epoch [{epoch+1}/{epochs}], "
              f"Train Loss: {total_loss:.4f}, "
              f"Train Acc: {correct_train / total_train:.4f}, "
              f"Val Acc: {correct_val / total_val:.4f}")

    # 保存模型
    torch.save(model.state_dict(), "rnn_position_model.pth")
    return model, vocab

# 预测函数
def predict(model, vocab, input_strings):
    """
    使用训练好的模型对输入字符串进行预测，并输出结果分析。
    """
    model.eval()
    with torch.no_grad():
        x = []
        for s in input_strings:
            ids = [vocab.get(c, vocab['unk']) for c in s]
            x.append(torch.LongTensor(ids))
        x = pad_sequence(x, batch_first=True)
        outputs = model(x)
        _, predicted = torch.max(outputs, 1)

        print("\n--- 测试结果分析 ---")
        for s, p in zip(input_strings, predicted):
            pos = int(p)
            # 判断是否预测正确
            result = '✅' if (s.find('a') == pos if 'a' in s else pos == 5) else '❌'
            desc = '无a' if pos == 5 else f'a出现在第{pos + 1}位'
            print(f"输入: {s}, 预测类别: {pos}, 含义: {desc}, 结果: {result}")

# 主程序入口
if __name__ == "__main__":
    model, vocab = train()
    test_strings = ["bcdea", "abcde", "edcba", "aaaaa", "bbbbb", "cdaeb"]
    predict(model, vocab, test_strings)
