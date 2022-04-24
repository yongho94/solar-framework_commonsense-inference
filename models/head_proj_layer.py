from torch import nn


class NonLinearHeadProjLayer(nn.Module):
    def __init__(self, input_hidden_size):
        super(NonLinearHeadProjLayer, self).__init__()
        self.linear_1 = nn.Linear(input_hidden_size, input_hidden_size)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(input_hidden_size, input_hidden_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class MultiHeadProjLayer(nn.Module):
    def __init__(self, input_hidden_size, head_num, head_size):
        super(MultiHeadProjLayer, self).__init__()
        self.linear_1 = nn.Linear(input_hidden_size, input_hidden_size)
        self.relu = nn.ReLU()
        self.head_size = head_size if head_size != -1 else int(input_hidden_size / head_num)
        self.multi_head_proj = nn.Linear(input_hidden_size, head_num * head_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.multi_head_proj(x)
        return x


class MultiHeadProjLayerDeeper(nn.Module):
    def __init__(self, input_hidden_size, head_num, head_size, inner_hidden_mul):
        super(MultiHeadProjLayerDeeper, self).__init__()
        self.linear_1 = nn.Linear(input_hidden_size, input_hidden_size * inner_hidden_mul)
        self.relu = nn.ReLU()
        self.head_size = head_size if head_size != -1 else int(input_hidden_size / head_num)
        self.linear_2 = nn.Linear(input_hidden_size * inner_hidden_mul, input_hidden_size)
        self.multi_head_proj = nn.Linear(input_hidden_size, head_num * head_size)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.multi_head_proj(x)
        return x