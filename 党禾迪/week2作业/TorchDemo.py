# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。

"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        # 增加一个隐藏层，提高模型的表达能力
        self.hidden = nn.Linear(input_size, 64)  # 新增隐藏层
        self.linear = nn.Linear(64, output_size)  # 输出维度调整为5
        self.activation = nn.Softmax(dim=1)  # 使用Softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    def forward(self, x, y=None):
        x = self.hidden(x)  # 新增隐藏层
        x = torch.relu(x)  # 使用ReLU激活函数
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, output_size)
        y_pred = self.activation(x)  # (batch_size, output_size) -> (batch_size, output_size)
        if y is not None:
            return self.loss(x, y)  # 注意：CrossEntropyLoss直接作用于未归一化的输出
        else:
            return y_pred  # 输出概率分布

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，返回向量及其最大值所在的维度（类别）
def build_sample():
    x = np.random.random(5)
    label = np.argmax(x)  # 最大值所在的维度作为类别
    return x, label

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签类型改为LongTensor
    # 修改为单个numpy数组转换为numpy数组
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y))

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    print("本次预测集中共有%d个样本" % test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        predicted_labels = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        for y_p, y_t in zip(predicted_labels, y):  # 与真实标签进行对比
            if y_p == y_t:
                correct += 1  # 预测正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 50
    batch_size = 32
    train_sample = 16000
    input_size = 5
    output_size = 5
    learning_rate = 0.0005
    # 建立模型
    model = TorchModel(input_size, output_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss  model.forward(x,y)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
        # 新增提前停止条件
        if acc >= 1.0:  # 浮点数比较使用>=更安全
            print("模型准确率达到100%，提前停止训练")
            break
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = TorchModel(input_size, output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        predicted_label = torch.argmax(res).item()  # 获取预测的类别
        actual_max_value = max(vec)  # 获取输入向量中的最大值
        actual_max_index = vec.index(actual_max_value)  # 获取最大值的索引
        is_correct = predicted_label == actual_max_index
        result_symbol = '✓' if is_correct else '✗'
        print("输入：%s, 预测值：%f, 实际最大值：%f, 实际最大值索引：%d, 预测类别：%d, 结果：%s" % (
            vec, res[predicted_label], actual_max_value, actual_max_index, predicted_label, result_symbol))  # 打印结果

if __name__ == "__main__":
    main()
    # 测试数据
    test_vec = [
        [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
        [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
        [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
        [0.09349776, 0.59416669, 0.42579291, 0.65567412, 0.1358894],
        [0.00499905, 0.00491005, 0.00491007, 0.00491005, 0.00499879],
    ]
    predict("model.bin", test_vec)
  
