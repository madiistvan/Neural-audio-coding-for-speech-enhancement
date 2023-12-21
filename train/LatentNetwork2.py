from torch import nn
class LatentNetwork(nn.Module):
    def __init__(self):
        super(LatentNetwork, self).__init__()

        self.conv_layer1 = nn.Conv1d(64, 1024, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(1024)        
        # Dilated convolutional layers for capturing long-range dependencies
        self.dilated_conv1 = nn.Conv1d(1024, 2048, kernel_size=5, stride=1, padding=5, dilation=5)

        self.elu = nn.ELU()
        self.batch_norm1 = nn.BatchNorm1d(2048)

        self.residual_conv1 = nn.Conv1d(2048, 2048, kernel_size=1, stride=1)  # Adjust channels if needed
        self.residual_batch_norm1 = nn.BatchNorm1d(2048)

        self.conv_layer4 = nn.Conv1d(2048, 1024, kernel_size=6, stride=1, padding=5)

        self.residual_conv2 = nn.Conv1d(1024, 1024, kernel_size=1, stride=1)  
        self.residual_batch_norm2 = nn.BatchNorm1d(1024)

        # Convolutional layer to match the output dimension
        self.conv_out = nn.Conv1d(1024, 64, kernel_size=6, stride=1, padding=4)

        #self.decoder = self.load_pretrained_model()

    def forward(self, x):
        x = self.elu(self.conv_layer1(x))
        x = self.bn1(x)

        x = self.elu(self.dilated_conv1(x))
        residual = self.residual_batch_norm1(self.residual_conv1(x))
        x = self.elu(x + residual)

        x = self.elu(self.conv_layer4(x))

        residual = self.residual_batch_norm2(self.residual_conv2(x))
        x = self.elu(x + residual)
        
        x = self.conv_out(x)
        
        return x