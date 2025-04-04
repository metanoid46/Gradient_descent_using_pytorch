# Import necessary libraries
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Function to display images from a batch of data
def imshow(img):
    # Unnormalize the image values (from the range [-1, 1] to [0, 1])
    img = img / 2 + 0.5     
    npimg = img.numpy()  # Convert the tensor to a NumPy array
    # Plot the image using matplotlib
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # Reorder dimensions for display (C, H, W) to (H, W, C)
    plt.show()


# Define the neural network class
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers of the network
        self.conv1 = nn.Conv2d(3, 6, 5)  # Convolutional layer (input channels = 3, output channels = 6, kernel size = 5)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer (kernel size = 2, stride = 2)
        self.conv2 = nn.Conv2d(6, 16, 5)  # Convolutional layer (input channels = 6, output channels = 16, kernel size = 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # Fully connected layer (input features = 16*5*5, output features = 120)
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer (input features = 120, output features = 84)
        self.fc3 = nn.Linear(84, 10)  # Output layer (input features = 84, output features = 10 for classification)

    def forward(self, x):
        # Define the forward pass of the network
        x = self.pool(F.relu(self.conv1(x)))  # Apply ReLU activation after the first convolution and max pooling
        x = self.pool(F.relu(self.conv2(x)))  # Apply ReLU activation after the second convolution and max pooling
        x = torch.flatten(x, 1)  # Flatten the output tensor (batch size, flattened features)
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first fully connected layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second fully connected layer
        x = self.fc3(x)  # Output layer (no activation here, as it's a classification problem with softmax later)
        return x


# Main function where the model training happens
if __name__ == '__main__':
    # Define the transformation for the images: convert to tensor and normalize
    transform = transforms.Compose(
        [transforms.ToTensor(),  # Convert images to tensor (PIL Image or numpy.ndarray to torch.Tensor)
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # Normalize the images to have a mean of 0.5 and std of 0.5

    batch_size = 4  # Batch size for training and testing

    # Load the CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Load the CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Define the class names for CIFAR-10 dataset
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Get some random training images
    dataiter = iter(trainloader)  # Create an iterator for the training data
    images, labels = next(dataiter)  # Get the first batch of images and their labels

    # Initialize the neural network
    net = Net()

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification tasks
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # Stochastic Gradient Descent optimizer

    # Training loop: iterate over the dataset multiple times (epochs)
    for epoch in range(2):  # Loop over the dataset 2 times (can increase for better performance)
        running_loss = 0.0  # Initialize running loss to track loss during training
        for i, data in enumerate(trainloader, 0):  # Loop over batches in the trainloader
            inputs, labels = data  # Get the inputs (images) and labels (true classes)

            # Zero the parameter gradients (clear previous gradients)
            optimizer.zero_grad()

            # Forward pass: compute the output of the network
            outputs = net(inputs)
            
            # Compute the loss (difference between predicted and true values)
            loss = criterion(outputs, labels)
            
            # Backward pass: compute gradients for the parameters
            loss.backward()
            
            # Update the parameters using the optimizer
            optimizer.step()

            # Accumulate the loss for printing every 2000 mini-batches
            running_loss += loss.item()
            if i % 2000 == 1999:    # Print the loss every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0  # Reset running loss

    print('Finished Training')

    # Save the trained model's state_dict (weights)
    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    # Get a batch of test images
    dataiter = iter(testloader)  # Create an iterator for the test data
    images, labels = next(dataiter)  # Get the first batch of images and their labels

    # Display the test images
    imshow(torchvision.utils.make_grid(images))  # Display the grid of images
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))  # Print the true labels

    # Load the trained model from the saved weights
    net = Net()  # Re-initialize the network
    net.load_state_dict(torch.load(PATH, weights_only=True))  # Load the weights into the network
