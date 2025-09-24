'''Train CIFAR10 with PyTorch.'''
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import math

from models import *


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()


    print(f"Epoch {epoch:03d} [{batch_idx+1:>4}/{len(trainloader)}] |"
                        f"lr={optimizer.param_groups[0]['lr']:.4g} | "
                        f"train_loss={train_loss:.4f} | train_acc={100.*correct/total:.3f}",
                        flush=True)

    return train_loss, 100.*correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print(f"Epoch {epoch:03d} |"
                            f"lr={optimizer.param_groups[0]['lr']:.4g} | "
                            f"test_loss={test_loss:.4f} | test_acc={100.*correct/total:.3f}",
                            flush=True)



    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

    return test_loss, acc


def plot_variable(trace: list, type: str, max_epoch: int, run_type: str):
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(range(max_epoch), trace)
    plt.xlabel('Epoch')
    plt.ylabel(f'{run_type} {type}')
    plt.title(f"{run_type} {type} Over Epochs for learning rate = {args.lr}")
    #plt.show()
    plt.savefig(f"{run_type}_{type}_trajectory_learning_rate_{args.lr}.png")
    plt.close()

max_epoch = start_epoch + 50
loss_trace = []
accuracy_trace = []
loss_trace_test = []
accuracy_trace_test = []

for epoch in range(start_epoch, max_epoch):
    loss, accuracy = train(epoch)
    loss_trace.append(loss)
    accuracy_trace.append(accuracy)

    loss_test, accuracy_test = test(epoch)
    loss_trace_test.append(loss_test)
    accuracy_trace_test.append(accuracy_test)
    scheduler.step()

# plotting traces
plot_variable(loss_trace, 'Loss', max_epoch, 'Training')
plot_variable(accuracy_trace, 'Accuracy', max_epoch, 'Training')
plot_variable(loss_trace_test, 'Loss', max_epoch, 'Testing')
plot_variable(accuracy_trace_test, 'Accuracy', max_epoch, 'Testing')
