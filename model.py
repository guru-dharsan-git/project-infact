import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# class transformers(nn.Module):
#     def __init__(self, input_size, hidden_size,hidden_size2, output_size):
#         super(transformers,self).__init__()
#         self.fc1 = nn.Linear(input_size,hidden_size)
#         self.fc2 = nn.Linear(hidden_size,hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2,output_size)

#     def forward(self,x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x=self.fc3(x)
#         return x