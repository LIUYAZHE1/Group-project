import torch.nn as nn
import torch.optim as optim
import pandas as pd
import torch
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class StockLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(StockLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层，增加了 dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # 线性层，先把 LSTM 的输出特征降维
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # 激活层 (ReLU)
        self.relu = nn.ReLU()
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
        # 批量归一化
        self.batch_norm = nn.BatchNorm1d(hidden_dim // 2)
        
        # 最后一层的输出层
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x):
        # 初始化隐藏状态和记忆单元
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM 层的输出
        out, _ = self.lstm(x, (h0, c0))
        
        # 选择最后一个时间步的输出
        out = out[:, -1, :]
        
        # 第一层全连接 + 激活 + dropout + 批归一化
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.batch_norm(out)
        
        # 最后一层全连接，输出最终结果
        out = self.fc2(out)
        
        return out

# 创建模型
def create_model(input_dim, hidden_dim=512, num_layers=2, output_dim=1):
    model = StockLSTM(input_dim, hidden_dim, num_layers, output_dim)
    return model

# 训练模型
def train_model(model, X_train, y_train, num_epochs=300, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    
    model.train()
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    return model, losses

# 绘制损失曲线
def plot_losses(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# 预测并反向缩放
def evaluate_model(model, X_train, X_test, y_train, y_test, scaler):
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train).detach().numpy()
        test_pred = model(X_test).detach().numpy()
    
    train_pred_scaled = scaler.inverse_transform(train_pred)
    test_pred_scaled = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.numpy())
    y_test_actual = scaler.inverse_transform(y_test.numpy())
    
    return train_pred_scaled, test_pred_scaled, y_train_actual, y_test_actual

# 绘制实际值与预测值
def plot_predictions(y_train_actual, y_test_actual, train_pred_scaled, test_pred_scaled):
    # 训练集实际值和预测值
    plt.figure(figsize=(10, 6))
    plt.plot(y_train_actual, label='Actual Train')
    plt.plot(train_pred_scaled, label='Predicted Train')
    plt.title('Training Set: Actual vs Predicted')
    plt.legend()
    plt.show()
    
    # 测试集实际值和预测值
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual, label='Actual Test')
    plt.plot(test_pred_scaled, label='Predicted Test')
    plt.title('Test Set: Actual vs Predicted')
    plt.legend()
    plt.show()

    # 测试集残差图
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_actual - test_pred_scaled, label='Residuals (Test)')
    plt.title('Residuals of Test Set')
    plt.xlabel('Time Steps')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()
    
    # 整体对比
    plt.figure(figsize=(12, 8))
    plt.plot(np.arange(len(y_train_actual)), y_train_actual, label='Actual Train')
    plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_test_actual)), y_test_actual, label='Actual Test')
    plt.plot(np.arange(len(y_train_actual)), train_pred_scaled, label='Predicted Train')
    plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_test_actual)), test_pred_scaled, label='Predicted Test')
    plt.title('Overall Actual vs Predicted (Train and Test)')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 读取并预处理数据
def load_and_preprocess_data(file_path, split_dates_ratios):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    data.set_index('Date', inplace=True)
    data = data.sort_index()
    return adjust_for_splits(data, split_dates_ratios)

# 股票拆股调整函数
def adjust_for_splits(data, split_dates_ratios):
    for split_date, ratio in split_dates_ratios.items():
        split_date = pd.Timestamp(split_date)
        data.loc[data.index < split_date, ['Open', 'High', 'Low', 'Close']] /= ratio
        data.loc[data.index < split_date, 'Volume'] *= ratio
    return data

# 计算回报率和夏普率
def calculate_sharpe_ratio(data, risk_free_rate=0.03):
    data['Returns'] = data['Close'].pct_change()
    data['Sharpe_Ratio'] = (data['Returns'].rolling(window=252).mean() - risk_free_rate) / data['Returns'].rolling(window=252).std()
    data.dropna(inplace=True)
    return data

# 标准化数据
def scale_data(data, features):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[features])
    return pd.DataFrame(scaled_data, columns=features, index=data.index), scaler

# 创建数据集
def create_dataset(tsla_data, time_step):
    x, y = [], []
    n_vectors = int(len(tsla_data) / (time_step + 1))
    for n in range(n_vectors):
        features = [tsla_data[col].values[n * (time_step + 1):n * (time_step + 1) + time_step] for col in tsla_data.columns]
        combined_features = np.column_stack(features)
        x.append(combined_features)
        y.append(tsla_data['Close'].values[n * (time_step + 1) + time_step])
    return np.array(x), np.array(y)

# 数据集分割
def split_dataset(X, y, split_ratio=0.8):
    split_index = int(split_ratio * len(X))
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# 绘制夏普率随时间变化
def plot_sharpe_ratio(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Sharpe_Ratio'])
    plt.title('Sharpe Ratio Over Time (Adjusted for Splits)')
    plt.xlabel('Date')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.show()


def data():
    set_seed(42)

    # 文件路径和拆股信息
    file_path = 'TSLA.csv'
    split_dates_ratios = {'2020-08-31': 5, '2022-08-25': 3}

    # 读取和预处理数据
    data = load_and_preprocess_data(file_path, split_dates_ratios)
    data = calculate_sharpe_ratio(data)

    # 标准化数据
    features = ['Open', 'High', 'Low', 'Volume', 'Close', 'Sharpe_Ratio']
    data_scaled, scaler = scale_data(data, features)

    # 创建时间步数据集
    time_step = 5
    X, y = create_dataset(data_scaled, time_step)

    # 转换为张量
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # 分割数据集
    X_train, X_test, y_train, y_test = split_dataset(X_tensor, y_tensor)

    # 为 'Close' 列单独创建一个 MinMaxScaler
    close_scaler = MinMaxScaler()
    close_scaler.fit(data[['Close']])

    # 绘制夏普率图
    plot_sharpe_ratio(data)

    # 输出数据形状
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # 确保 return 语句正确缩进
    return X_train, X_test, y_train, y_test, close_scaler


X_train, X_test, y_train, y_test, close_scaler = data()

file_path = 'TSLA.csv'
data = pd.read_csv(file_path, parse_dates=['Date'])
data.set_index('Date', inplace=True)


# 修改后的 LSTM 函数
def LSTM(X_train, X_test, y_train, y_test, scaler, input_dim):
    model = create_model(input_dim)
    model, losses = train_model(model, X_train, y_train)
    plot_losses(losses)

    # 将模型返回，用于后续评估
    return model


# 调用主函数
close_scaler = MinMaxScaler()
close_scaler.fit(data[['Close']])  # 为 'Close' 列单独创建一个 MinMaxScaler

# 调用 LSTM 函数并获取模型
model = LSTM(X_train, X_test, y_train, y_test, close_scaler, X_train.shape[2])

# 用训练好的 model 进行评估
train_pred_scaled, test_pred_scaled, y_train_actual, y_test_actual = evaluate_model(model, X_train, X_test, y_train,
                                                                                    y_test, close_scaler)
plot_predictions(y_train_actual, y_test_actual, train_pred_scaled, test_pred_scaled)


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 计算评价指标的函数
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mae, r2

# 评估训练集的性能
mse_train, rmse_train, mae_train, r2_train = evaluate_model(y_train_actual, train_pred_scaled)
print(f'Training Set Performance:')
print(f'MSE: {mse_train:.4f}, RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}, R²: {r2_train:.4f}')

# 评估测试集的性能
mse_test, rmse_test, mae_test, r2_test = evaluate_model(y_test_actual, test_pred_scaled)
print(f'\nTest Set Performance:')
print(f'MSE: {mse_test:.4f}, RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}, R²: {r2_test:.4f}')

