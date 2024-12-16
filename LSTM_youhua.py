import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from skopt import gp_minimize
from skopt.space import Real, Integer
import spxy
import pro
# 读取数据
#file_path = 'F:\光谱数据\模型训练数据\奥普天成 苹果数据//苹果光谱数据_分类276 - 副本.csv'
file_path ='F:\光谱数据\模型训练数据\论文—苹果光谱数据//洛川苹果红富士分类.csv'
data = pd.read_csv(file_path)

# 数据预处理
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_SNV=pro.SNV(X)
X_train, X_test, y_train, y_test = spxy.spxy(X_SNV, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, 1, X_train.shape[1])
X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1, X_test.shape[1])
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

import torch
import torch.nn as nn

# 定义多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert input_size % num_heads == 0, "Input size must be divisible by number of heads"
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads

        self.query = nn.Linear(input_size, input_size)
        self.key = nn.Linear(input_size, input_size)
        self.value = nn.Linear(input_size, input_size)
        self.fc_out = nn.Linear(input_size, input_size)

    def forward(self, x):
        N, seq_len, embed_size = x.shape
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Split the embedding into multiple heads
        Q = Q.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention = torch.einsum("nqhd,nkhd->nhqk", [Q, K]) / (self.head_dim ** (1 / 2))
        attention = torch.softmax(attention, dim=-1)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention, V]).reshape(N, seq_len, embed_size)

        out = self.fc_out(out)
        return out

# 定义双向LSTM + 多头注意力模型
class BiLSTM_AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate, num_heads):
        super(BiLSTM_AttentionModel, self).__init__()
        if (hidden_size * 2) % num_heads != 0:
            # 自动调整 num_heads，确保其为 input_size 的因数
            num_heads = 1  # 你可以选择 1 或其他合适的值
            print(f"Warning: Adjusted num_heads to {num_heads} to be divisible by input size.")
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.attention = MultiHeadAttention(hidden_size * 2, num_heads)  # 双向LSTM输出维度是hidden_size的2倍
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h, _ = self.lstm(x)
        h = self.attention(h)
        h = self.dropout(h[:, -1, :])
        out = self.fc(h)
        return self.sigmoid(out)


# 定义训练和评估函数
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            predicted = (output.squeeze() > 0.5).float()
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

# 贝叶斯优化
def objective(params):
    hidden_size, dropout_rate, learning_rate, batch_size, epochs, num_heads = params
    batch_size = int(batch_size)  # 确保 batch_size 是整数
    model = BiLSTM_AttentionModel(input_size=X_train.shape[2], hidden_size=hidden_size, dropout_rate=dropout_rate, num_heads=num_heads)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    train_model(model, train_loader, criterion, optimizer, epochs)
    accuracy = evaluate_model(model, test_loader)

    return -accuracy  # 贝叶斯优化的目标是最小化

param_space = [
    Integer(10, 100, name='hidden_size'),
    Real(0.1, 0.5, name='dropout_rate'),
    Real(1e-4, 1e-2, 'log-uniform', name='learning_rate'),
    Integer(16, 128, name='batch_size'),
    Integer(10, 1000 ,name='epochs'),
    Integer(1, 4, name='num_heads')  # 确保包含 num_heads 参数
]

# 使用贝叶斯优化
res = gp_minimize(objective, param_space, n_calls=20, random_state=42)

# 输出最佳参数
best_params = {
    'hidden_size': res.x[0],
    'dropout_rate': res.x[1],
    'learning_rate': res.x[2],
    'batch_size': res.x[3],
    'epochs': res.x[4],
    'num_heads': res.x[5]
}
print("Best parameters found: ", best_params)

# 使用最佳参数评估模型
best_model = BiLSTM_AttentionModel(input_size=X_train.shape[2], hidden_size=best_params['hidden_size'], dropout_rate=best_params['dropout_rate'],num_heads=best_params['num_heads'] )
best_optimizer = optim.Adam(best_model.parameters(), lr=best_params['learning_rate'])
best_criterion = nn.BCELoss()

# 将数据和模型转移到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)
X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)
best_params['batch_size'] = int(best_params['batch_size'])
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=best_params['batch_size'], shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=best_params['batch_size'], shuffle=False)

# 训练和评估最佳模型
train_model(best_model, train_loader, best_criterion, best_optimizer, best_params['epochs'])
accuracy = evaluate_model(best_model, test_loader)
print(f"Test Accuracy: {accuracy:.4f}")
