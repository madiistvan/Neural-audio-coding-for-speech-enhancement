import torch.nn as nn

class LatentNetwork(nn.Module):
    def __init__(self):
        super(LatentNetwork, self).__init__()

        self.conv_layer1 = nn.Conv1d(64, 1024, kernel_size=3, stride=1, padding=2)
        nn.init.kaiming_normal_(self.conv_layer1.weight, nonlinearity='leaky_relu')
        
        self.bn1 = nn.BatchNorm1d(1024)        

        self.dilated_conv1 = nn.Conv1d(1024, 2048, kernel_size=5, stride=1, padding=5, dilation=5)
        nn.init.kaiming_normal_(self.dilated_conv1.weight, nonlinearity='leaky_relu')

        self.activation = nn.ELU()
        self.bn2 = nn.BatchNorm1d(2048)

        self.residual_conv1 = nn.Conv1d(2048, 2048, kernel_size=1, stride=1) 
        nn.init.kaiming_normal_(self.residual_conv1.weight, nonlinearity='leaky_relu')
 
        self.residual_bn1 = nn.BatchNorm1d(2048)

        self.conv_layer4 = nn.Conv1d(2048, 1024, kernel_size=6, stride=1, padding=5)
        nn.init.kaiming_normal_(self.conv_layer4.weight, nonlinearity='leaky_relu')
        self.bn4 = nn.BatchNorm1d(1024)


        self.residual_conv2 = nn.Conv1d(1024, 1024, kernel_size=1, stride=1)
        nn.init.kaiming_normal_(self.residual_conv2.weight, nonlinearity='leaky_relu')
  
        self.residual_bn2 = nn.BatchNorm1d(1024)

        self.conv_out = nn.Conv1d(1024, 64, kernel_size=6, stride=1, padding=4)
        nn.init.kaiming_normal_(self.conv_out.weight, nonlinearity='leaky_relu')



    def forward(self, x):
        x = self.activation(self.conv_layer1(x))
        x = self.bn1(x)

        x = self.dilated_conv1(x)

        residual = self.residual_conv1(self.bn2(self.activation(x)))
        x = self.activation(x + residual)
        x = self.residual_bn1(x)

        x = self.conv_layer4(x)

        residual = self.residual_conv2(self.bn4(self.activation(x)))
        x = self.activation(x + residual)
        x = self.residual_bn2(x)

        x = self.conv_out(x)
        
        return x
