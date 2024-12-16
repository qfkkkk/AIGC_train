import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import spxy
import pro

# Load the dataset
file_path = 'F:\\光谱数据\\模型训练数据\\论文—苹果光谱数据\\洛川苹果红富士回归.csv'
data = pd.read_csv(file_path)

# Prepare the data
X = data.iloc[:, :-1].values  # All columns except the last one are features
y = data.iloc[:, -1].values  # The last column is the target

# Normalize the features
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)

# Reshape the data to fit LSTM input requirements: [samples, time steps, features]
X_scaled = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
X_SG = pro.SNV(X_scaled)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = spxy.spxy(X_SG, y, test_size=0.2)

# Convert the data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Define the LSTM model with MultiHeadAttention
class BiLSTM_Attention_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_heads):
        super(BiLSTM_Attention_Model, self).__init__()
        # Define the BiLSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Multihead Attention
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=num_heads, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size * 2, 1)  # 2 for bidirectional LSTM

    def forward(self, x):
        # Initialize hidden state and cell state
        h_0 = torch.zeros(2 * 2, x.size(0), 64)  # 2 for bidirectional, 2 for layers
        c_0 = torch.zeros(2 * 2, x.size(0), 64)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h_0, c_0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)

        # Apply attention mechanism
        attn_output, _ = self.attention(out, out, out)

        # Pass through the fully connected layer (using the output from the last time step)
        out = self.fc(attn_output[:, -1, :])  # Use the last time step's output
        return out


# Initialize the model, define the loss function and the optimizer
input_size = 1
hidden_size = 64
num_layers = 2
num_heads = 4
model = BiLSTM_Attention_Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               num_heads=num_heads)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 64
for epoch in range(num_epochs):
    model.train()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Predict on the test set
model.eval()
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy()

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)
print(f'R^2 score: {r2}')
