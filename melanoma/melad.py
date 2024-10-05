import torch
import torch.nn as nn
import torch.nn.functional as F

class MelaD(nn.Module):
    def __init__(self):
        super(MelaD, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.conv5 = nn.Conv2d(64, 64, 3, padding=8, dilation=8)
        self.conv6 = nn.Conv2d(64, 64, 3, padding=16, dilation=16)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)

        # Batch normalization layers
        self.batch_norm1 = nn.BatchNorm2d(128)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.batch_norm4 = nn.BatchNorm2d(64)
        self.batch_norm5 = nn.BatchNorm2d(64)
        self.batch_norm6 = nn.BatchNorm2d(64)
        self.batch_norm7 = nn.BatchNorm2d(64)

        # Average pooling layer
        # self.avg_pool = nn.AvgPool2d(kernel_size=(7, 7))  # Adjust kernel size as necessary

        # Final fully connected layer
        self.fc = nn.Linear(64, 2)  # Adjust this if necessary based on the output shape

    def forward(self, x):
        # Layer 1
        x = F.relu(self.conv1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm1(x)
        

        # Layer 2
        x = F.relu(self.conv2(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm2(x)
        

        # Layer 3
        x = F.relu(self.conv3(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm3(x)
        

        # Layer 4
        x = F.relu(self.conv4(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm4(x)
        

        # Layer 5
        x = F.relu(self.conv5(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm5(x)
        

        # Layer 6
        x = F.relu(self.conv6(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm6(x)
        

        # Layer 7
        x = F.relu(self.conv7(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.batch_norm7(x)
        

        # Layer 8
        x = F.relu(self.conv8(x))

        # Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))  # This will reduce the spatial dimensions

        # Pass through the fully connected layer without flattening
        x = x.view(x.size(0), -1)  # Use view to reshape for the Linear layer

        # Output layer
        x = self.fc(x)  # Final output layer

        # Softmax activation for output
        # return F.softmax(x, dim=1)
        return x

# Instantiate the model
# model = MeshNetTest()

# # Test the method with a sample input
# input_data = torch.randn(1, 3, 150, 150)  # Example input tensor with batch size 1 and image size 150x150
# output = model.meshnet_test(input_data)
# print(output)
