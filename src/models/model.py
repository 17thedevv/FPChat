import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        # Lớp ẩn thứ nhất
        self.l1 = nn.Linear(input_size, hidden_size) 
        # Lớp ẩn thứ hai
        self.l2 = nn.Linear(hidden_size, hidden_size) 
        # Lớp đầu ra (số chiều bằng số lượng Intent tags)
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Hàm kích hoạt phi tuyến
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        # Không dùng activation/softmax ở cuối vì PyTorch CrossEntropyLoss đã tích hợp sẵn
        return out