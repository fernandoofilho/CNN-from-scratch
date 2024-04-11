import torch 
import torch.nn as nn
import torch.nn.functional as functional


# CNN architecture
# init as a subclass of nn.Module


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input layer 
        self.conv1 =  nn.Conv2d(3,6,5)
        
        #pooling - kernel 2x2 with 2 steps
        self.pool1 = nn.MaxPool2d(2,2)
        
        # conv layer  
        self.conv2 = nn.Conv2d(6,16,5) 
        
        # max pooling - 2x2 kernel with 2 steps
        self.pool2 = nn.MaxPool2d(2,2)
        
        # Fully connected conv layer
        # Linear transform into 120-dimensional space
        self.fc1 = nn.Linear(16*5*5, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # 10 = len(classes)
        
        
    def forward(self, x):
        # first conv layer with ReLU activation + maxpooling
        x = self.conv1(x)
        x = functional.relu(x)
        x = self.pool1(x)
        
        # second layer with ReLU activation + maxpooling
        x = self.conv2(x)
        x = functional.relu(x)
        x = self.pool2(x)
        
        # Flattened layer 
        x = x.view(-1, 16*5*5)
        
        # fully connected layer with ReLU activation
        x = self.fc1(x)
        x = functional.relu(x)
        
        
        # fully connected layer with ReLU activation
        x = self.fc2(x)
        x = functional.relu(x)
        
        # output
        x = self.fc3(x)
        
        return x
         