import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset


# ToTensor()는 이미지의 픽셀 값 범위를 0~1로 조정
# normalize 하는 이유 -> 오차역전파시 gradient계산 수행 -> 데이터가 유사한 범위를 가지도록 하기 위함 -> 수렴 속도가 더 빨라진다(cost function이 타원보다 구에 가까움)
# https://light-tree.tistory.com/132
# transofrms.Normalize((R채널 평균, G채널 평균, B채널 평균), (R채널 표준편차, G채널 표준편차, B채널 표준편차))
# 각 채널 별 평균을 뺀 후 표준편차로 나누어 계산
# 아래 예시에서는 -1 ~ 1로 변환

# 데이터셋 instance 생성
# 데이터를 저장하려는 파일시스템 경로, 학습용 여부, 다운로우 여부, transform 객체
# 무작위 추출한 4개의 batch image를 trainset에서 추출
# num workers => 복수 개의 프로세스
#cifar10 training 50000 test 10000

def load_dataset(dataset, batch_size):
    transform_train = transforms.Compose([ 
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32,32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    transform_test = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))])

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root = './data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root = './data/cifar10', train=False, download=True, transform=transform_test)
        num_classes = 10
    elif dataset == 'cifar100':
        trainset = torchvision.datasets.CIFAR10(root = './data/cifar10', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root = './data/cifar10', train=False, download=True, transform=transform_test)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader,testloader,num_classes