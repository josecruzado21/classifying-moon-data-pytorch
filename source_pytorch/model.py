import torch
import torch.nn as nn #Pytorch nn classes help us train Neural networks in a more concise way. We end up with shorter, understandable and flexible code.
import torch.nn.functional as F #Functional permits us to work with built-in activation and loss function

class MLP(nn.Module):
    #nn.Module is the base class for all neural network modules
    def __init__(self,dim_input,dim_hidden,dim_output):
        '''s
        Inputs:
            dim_input: Number of inputs (dimension of the input layer)
            dim_hidden: Dimension of the hiddel layer(s)
            dim_output: Number of outputs
            
        '''
        super(MLP,self).__init__()
        #creating fully connected layers
        self.fc1=nn.Linear(dim_input,dim_hidden)
        self.fc2=nn.Linear(dim_hidden,dim_output)
        self.dropout=nn.Dropout(0.3)
        self.sigmoid=nn.Sigmoid()
        
    def forward(self,x):
        '''
        Feedforward behavior of the nn.
        Input:
            x: batch of input features
        Returns:
            Activated sigmoid function
        '''
        
        out=F.relu(self.fc1(x))
        out=self.dropout(out)
        out=self.fc2(out)
        
        return self.sigmoid(out)
        