import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import datetime

import dataloader
import model

from torch.utils.tensorboard import SummaryWriter


def parsing_argument():
    parser = argparse.ArgumentParser(description="argparse_test")

    parser.add_argument('-e', '--epochs', metavar='int', type=int, help='epochs', default=2)
    parser.add_argument('-lr', '--learningrate', metavar='float', type=float, help='lr', default=0.001)
    parser.add_argument('-bs', '--batchsize', metavar='int', type=int, help='batchsize', default=4)
    parser.add_argument('-d', '--dataset', metavar='str', type=str, help='dataset [cifar10, cifar100]', default='cifar10')
    parser.add_argument('-opt', '--optimizer', metavar='str', type=str, help='optimizer [adam, sgd]', default='sgd')
    parser.add_argument('-m', '--model', metavar='str', type=str, help='models [LeNet, ResNet18]', default='LeNet')
    parser.add_argument('-tr', '--train', help='train', action='store_true')
    parser.add_argument('-tc', '--test', help='test', action='store_true')
    parser.add_argument('-p', '--patience', type=int, help='patience', default=10)

    return parser.parse_args()

def train_per_epoch(dataloader, net, optimizer, criterion, device):
    net.train()
    running_loss = 0.0

    for i,data in enumerate(dataloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss
    
def test(dataloader, net, criterion, device):
    total = 0
    correct = 0
    loss = 0
    
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            ouputs = net(images)
            tmp_loss = criterion(ouputs, labels)
            loss += tmp_loss.item() * images.size(0)
            _, predicted = torch.max(ouputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100*correct/total
    return loss,accuracy

def train(args, trainloader, net, optimizer, scheduler, criterion, device, writer, outputs_log):
    for epoch in range(args.epochs):
        running_loss = train_per_epoch(trainloader, net, optimizer, criterion, device)
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss/100))
        print('epoch[%d] - training loss : %.3f'%(epoch+1, running_loss/100),file=outputs_log)
        
        writer.add_scalar('train / train_loss', running_loss/100, epoch) 
        running_loss = 0.0

        if scheduler:
            scheduler.step()
    torch.save(net.state_dict(), './model_state_dict.pt')
    print('Finished Training',file=outputs_log)

def set_parameter(args, net):
    if args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr = args.lr)
        scheduler = None
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    return optimizer,scheduler,criterion

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = parsing_argument()
    cur_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    
    outputs_log = open(f'outputs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{cur_time}.txt','w')
    writer = SummaryWriter(f'logs/{args.model}_{args.epochs}ep_{args.learningrate}lr_{args.optimizer}_{cur_time}')

    trainloader,testloader,num_classes = dataloader.load_dataset(args.dataset, args.batchsize)
    net = model.set_net(args.model, num_classes)
    net.to(device)
    optimizer,scheduler,criterion = set_parameter(args, net)
    
    if args.train:
        print(f"Training start ...")
        train(args, trainloader, net, optimizer, scheduler, criterion, device, writer, outputs_log)
    if args.test:
        print(f"Test start ...")
        loss,acc = test(testloader, net, criterion, device)
        print("accuracy: %.3f %%"%acc)
        print("accuracy: %.3f %%"%acc, file=outputs_log)
        
    outputs_log.close()
    writer.close()

if __name__ == "__main__":
    main()
