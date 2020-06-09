import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import sklearn.metrics as skmetrics
import seaborn as sns

from classification_dataset import ClassificationDataset

import sys
import argparse

def compute_accuracy(net, testloader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def confusion_matrix(net, testloader, device):
    net.eval()
    true_labels = []
    predictions = []
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            predicted = torch.argmax(outputs.data, 1)
            true_labels.append(labels.cpu().numpy())
            predictions.append(predicted.cpu().numpy())
    true_labels = np.hstack(true_labels)
    predictions = np.hstack(predictions)

    return skmetrics.confusion_matrix(true_labels, predictions)

def train(train_path, test_path, weightfile):
    # choose net
    net = resnet18(num_classes = 3)
    device = torch.device('cuda')
    net.to(device)

    # train transforms (NOTE: some transforms are included in the ClassificationDataset class)
    transform_train = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ColorJitter(0.1,0.1,0.1,0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4786, 0.4712, 0.4665), (0.2352, 0.2317, 0.2367))
    ])

    # test transforms
    transform_test = transforms.Compose([
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize((0.4786, 0.4712, 0.4665), (0.2352, 0.2317, 0.2367)),
    ])

    # initialize dataloaders
    dataset_train = ClassificationDataset(train_path, transform_train, True)
    trainloader = torch.utils.data.DataLoader(dataset = dataset_train, batch_size = 128, shuffle = True)

    dataset_test = ClassificationDataset(test_path, transform_test, False)
    testloader = torch.utils.data.DataLoader(dataset = dataset_test, batch_size = 128, shuffle = False)

    # initialize SGD and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = 0.1, momentum = 0.9, weight_decay = 5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 15, gamma = 0.1)

    epochs = 60

    for epoch in range(epochs):
        running_loss = 0.0
        net.train()
        for i, (inputs, labels) in enumerate(trainloader, 0):
            # Transfer to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        test_loss = compute_accuracy(net, testloader, device)
        print('[%d, %5d] loss: %.3f test accuracy: %.3f' % (epoch+1, i+1, running_loss, test_loss))
        running_loss = 0.0
        scheduler.step()

    accuracy = compute_accuracy(net, testloader, device)
    print('Accuracy of the network on the test images: %.3f' % accuracy)

    # confusion matrix
    c_matrix = confusion_matrix(net, testloader)
    print(c_matrix)

    torch.save(net.state_dict(), weightfile)
    print('Model saved to %s' % weightfile)

ap = argparse.ArgumentParser()
ap.add_argument("-tr", "--train_path", help="path to training csv file")
ap.add_argument("-te", "--test_path", help="path to testing csv file")
ap.add_argument("-w", "--weights", help="weightfile to be saved")
args = vars(ap.parse_args())

if not args.get("train_path", False):
        print("No train path provided to video")
        sys.exit()
if not args.get("test_path", False):
        print("No test path provided to video")
        sys.exit()
if not args.get("weights", False):
        print("No weightfile provided")
        sys.exit()

train(args.get("train_path"), args.get("test_path"), args.get("weights"))
