import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(f'Using device: {device}')
# Hyper Parameters:
num_epochs = 15
batch_size = 50
learning_rate = 0.002

print('Preparing dataset! Making augmentations...')
trans_1 = transforms.Compose([
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(hue=0.4),
    transforms.RandomHorizontalFlip(p=1),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

trans_2 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])
trans_3 = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomAffine(10, translate=None, scale=None, shear=None),
    transforms.ColorJitter(saturation=0.25),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.247, 0.2434, 0.2615)),
])

print('Concatenating datasets....')
train_dataset = torch.utils.data.ConcatDataset(
    [datasets.CIFAR10(root='./data/', train=True, transform=trans_1, download=True),
     datasets.CIFAR10(root='./data/', train=True, transform=trans_2, download=True),
     datasets.CIFAR10(root='./data/', train=True, transform=trans_3, download=True)])

test_dataset = datasets.CIFAR10(root='./data/',
                                train=False,
                                transform=transform_test,
                                download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layers = nn.Sequential(nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, padding=2),
            nn.BatchNorm2d(20),
            nn.PReLU(),
        ), nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=5, padding=2),
            nn.PReLU(),
        ), nn.Sequential(
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.BatchNorm2d(20),
            nn.PReLU(),
            nn.MaxPool2d(2),
        ), nn.Sequential(
            nn.Conv2d(20, 24, kernel_size=3, padding=1),
            nn.PReLU(),
        ), nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.PReLU(),
            nn.MaxPool2d(2),
        ), nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        ))
        self.fully_connected = nn.Linear(1024, 10)
        self.dropout = nn.Dropout(p=0.25)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fully_connected(out)
        return self.logsoftmax(out)


def plot_epochs(train_loss, train_acc, test_loss, test_acc):
    import matplotlib.pyplot as plt
    # Loss plot
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='test')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Error plot
    plt.plot([100 - x for x in train_acc], label='train')
    plt.plot([100 - x for x in test_acc], label='test')
    plt.title('Error over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()


model = ConvNet()

print(f'Num of trainable parameters: [{count_parameters(model)}/50000]')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_model_q1():
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    print('Training...')
    for epoch in range(num_epochs):
        running_loss = 0.0
        n_samples, n_correct = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = 100 * n_correct / n_samples
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | Train Loss: {running_loss / len(train_loader):.4f} | Acc: {acc: .4f}")
        train_loss.append(running_loss / len(train_loader))
        train_acc.append(acc)

        # eval
        running_loss = 0.0
        with torch.no_grad():
            n_samples, n_correct = 0, 0
            n_class_correct = [0 for _ in range(10)]
            n_class_sample = [0 for _ in range(10)]
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                for i in range(batch_size):
                    label = labels[i]
                    pred = predicted[i]
                    if (label == pred):
                        n_class_correct[label] += 1
                    n_class_sample[label] += 1
            acc = 100 * n_correct / n_samples
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | Test Loss: {running_loss / len(test_loader):.4f} | Acc: {acc: .4f}")
            test_loss.append(running_loss / len(test_loader))
            test_acc.append(acc)
            print('-' * 50)
    print("Finished training")
    torch.save(model, 'model_q1.pth')
    plot_epochs(train_loss, train_acc, test_loss, test_acc)


def evaluate_model_q1():
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform_test, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = torch.load('model_q1.pth')
    with torch.no_grad():
        n_samples, n_correct = 0, 0
        n_class_correct = [0 for i in range(10)]
        n_class_sample = [0 for i in range(10)]
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_sample[label] += 1
        acc = 100 * n_correct / n_samples
        print(f"Accuracy on the test set: {acc}%")


if __name__ == '__main__':
    train_model_q1()
    evaluate_model_q1()
