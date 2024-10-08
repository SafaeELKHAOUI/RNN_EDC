import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class ECGDataset(Dataset):
    def __init__(self, mat_files):
        self.mat_files = mat_files

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        mat_file_path = self.mat_files[idx]
        data = scipy.io.loadmat(mat_file_path)
        ecg_signal = data['val'][0]  # Extract the ECG signal
        ecg_tensor = torch.tensor(ecg_signal, dtype=torch.float32)
        ecg_tensor = ecg_tensor.unsqueeze(0)  # Add channel dimension [1, signal_length]
        return ecg_tensor

# Example: Load a subset of .mat files
subset_files = ['C:/Users/elkha/Downloads/training2017/training2017/A08516.mat', 'C:/Users/elkha/Downloads/training2017/training2017/A08517.mat']  # List a few .mat files
dataset = ECGDataset(subset_files)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(32 * (9000 - 4), 64)  # Adjust sequence length
        self.fc2 = nn.Linear(64, 8528)  # Adjust num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 8528)  # Adjust num_classes

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]  # Use the last time-step output
        x = self.fc(x)
        return x

def train_model(model, dataloader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for data in dataloader:  # No labels
            optimizer.zero_grad()
            outputs = model(data)
            # You need a criterion if labels are present
            # loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            # running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{epochs}] completed')


# Example: Train CNN model
cnn_model = CNNModel()
train_model(cnn_model, dataloader)

# Train RNN model similarly
rnn_model = RNNModel()
train_model(rnn_model, dataloader)




