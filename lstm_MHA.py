import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import spxy
import pro

# 检查是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 数据预处理
file_path = 'F:\\光谱数据\\模型训练数据\\论文—苹果光谱数据//洛川苹果红富士分类.csv'
data = pd.read_csv(file_path)

X = data.iloc[:, :-1].values  # 提取特征列
y = data.iloc[:, -1].values  # 提取标签列
scaler = StandardScaler()
X_scaler = scaler.fit_transform(X)
X_SG = pro.SNV(X_scaler)
X_train, X_test, y_train, y_test = spxy.spxy(X_SG, y,test_size=0.2)  # 根据实际情况修改

# 将数据转换为Tensor格式并移动到GPU
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, X_train.shape[1]).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, X_test.shape[1]).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 创建 DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)


# 定义多头注意力机制的BiLSTM模型
class BiLSTM_MultiheadAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, output_size=3):
        super(BiLSTM_MultiheadAttentionModel, self).__init__()

        # 双向LSTM层
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        # self.lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, batch_first=True, bidirectional=True)

        # 多头注意力机制
        self.attention1 = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)  # 用于LSTM1输出
        # self.attention2 = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)  # 用于LSTM2输出

        # Dropout层
        self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.25)

        # 全连接层
        self.fc4 = nn.Linear(hidden_size * 2, output_size)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM1 + Dropout1
        x, _ = self.lstm1(x)
        attn_output1, _ = self.attention1(x, x, x)  # 多头注意力机制
        x = self.dropout1(attn_output1)
        x = self.relu(x)

        # # LSTM2 + Dropout2
        # x, _ = self.lstm2(x)
        # attn_output2, _ = self.attention2(x, x, x)  # 多头注意力机制
        # x = self.dropout2(attn_output2)
        # x = self.relu(x)

        # 取最后时间步的输出
        x = x[:, -1, :]

        # 全连接层
        x = self.fc4(x)
        # x = F.softmax(x, dim=1)
        return x

# 定义模型、损失函数、优化器
input_size = X_train.shape[1]  # 输入维度
hidden_size = 64  # 隐藏层大小
num_heads = 4  # 多头注意力机制中的头数

model = BiLSTM_MultiheadAttentionModel(input_size=input_size, hidden_size=hidden_size, num_heads=num_heads).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.0005, patience=20, min_lr=1e-6)


# 准确率计算函数
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total


# 获取模型预测和真实标签
def get_predictions_and_labels(model, data_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    return all_labels, all_preds

# 打印分类报告
def print_classification_report(model, data_loader, classes):
    true_labels, predicted_labels = get_predictions_and_labels(model, data_loader)
    report = classification_report(true_labels, predicted_labels, target_names=classes)
    print("Classification Report:\n")
    print(report)

# 损失曲线和混淆矩阵绘制函数
def plot_confusion_matrix(model, data_loader, classes):
    true_labels, predicted_labels = get_predictions_and_labels(model, data_loader)
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()

# 损失曲线的绘制
def plot_loss_curve(train_losses, test_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(test_losses) + 1), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.show()

# 训练模型函数，记录训练集和验证集损失并绘制损失曲线
def train_model(model, train_loader, test_loader, epochs):
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()

        # 记录训练集的损失
        train_losses.append(running_train_loss / len(train_loader))

        # 计算验证集损失
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_test_loss += loss.item()

        # 记录验证集的损失
        test_losses.append(running_test_loss / len(test_loader))

        # 更新学习率调度器
        scheduler.step(running_train_loss)

        # 计算训练集和测试集准确率
        train_accuracy = calculate_accuracy(model, train_loader)
        test_accuracy = calculate_accuracy(model, test_loader)

        # 打印每个epoch的损失和准确率
        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_train_loss:.4f}, Test Loss: {running_test_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')

    # 绘制训练集和验证集的损失曲线
    plot_loss_curve(train_losses, test_losses)

    # 打印分类报告
    print_classification_report(model, test_loader, classes=['Class 0', 'Class 1', 'Class 2'])

    # 绘制混淆矩阵
    plot_confusion_matrix(model, test_loader, classes=['Class 0', 'Class 1', 'Class 2'])


# 训练模型并绘制损失曲线、混淆矩阵和打印分类报告
train_model(model, train_loader, test_loader, epochs=500)

