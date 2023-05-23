import torch 
from PIL import Image
from tqdm import *
import torch.nn.functional as F
from torchsummary import summary
from torchvision import transforms
from torchvision import datasets
from torch.utils import data
from torch import nn
import matplotlib.pyplot as plt
import torchvision

BATCH_SIZE = 20
LR = 0.00001
EPOCH = 200
split_ratio = [0.6, 0.2, 0.2]
split_names = ['train', 'eval', 'test']
split_cnt = [0,0,0]
transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset_train = datasets.ImageFolder('train', transform)
dataset_test = datasets.ImageFolder('validation', transform)

train_loader = data.DataLoader(
    dataset=dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True)

test_loader = data.DataLoader(
    dataset=dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 3)
        self.conv1_2 = nn.Conv2d(32, 64, 3)
        self.max_pool1 = nn.MaxPool2d(2)
        self.conv2_1 = nn.Conv2d(64, 128, 3)
        self.conv2_2 = nn.Conv2d(128, 128, 3)
        self.max_pool2 = nn.MaxPool2d(2)
        self.conv3_1 = nn.Conv2d(128, 256, 3)
        self.conv3_2 = nn.Conv2d(256, 256, 3)
        self.conv3_3 = nn.Conv2d(256, 256, 3)
        self.conv3_4 = nn.Conv2d(256, 256, 3)
        self.max_pool3 = nn.MaxPool2d(2)
        self.conv4_1 = nn.Conv2d(256, 512, 3)
        self.conv4_2 = nn.Conv2d(512, 512, 3)
        self.conv4_3 = nn.Conv2d(512, 512, 3)
        self.conv4_4 = nn.Conv2d(512, 512, 3)
        self.max_pool4 = nn.MaxPool2d(2)
        self.conv5_1 = nn.Conv2d(512, 512, 3)
        self.conv5_2 = nn.Conv2d(512, 512, 3)
        self.conv5_3 = nn.Conv2d(512, 512, 3)
        self.conv5_4 = nn.Conv2d(512, 512, 3)
        self.max_pool5 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(6400,512)
        self.fc1.weight.data.normal_(0, 00.1)
        self.fc2 = nn.Linear(512, 1)
        self.fc2.weight.data.normal_(0, 00.1)
        self.dropout = nn.Dropout(p=0.5)
        self.BN = nn.BatchNorm2d(256)
    def forward(self,x):
        in_size = x.size(0)
        x = self.conv1_1(x)
        x = F.relu(x)
        x = self.conv1_2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        
        x = self.conv2_1(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv2_2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        
        x = self.conv3_1(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3_2(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        
        x = x.view(in_size, -1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x
    
    # def forward(self,x):
    #     in_size = x.size(0)
    #     x = self.conv1_1(x)
    #     x = F.relu(x)
    #     x = self.max_pool1(x)
    #     x = self.conv1_2(x)
    #     x = F.relu(x)
    #     x = self.max_pool2(x)
    
        
    #     x = x.view(in_size, -1)
    #     x = self.fc1(x)
    #     x = F.relu(x)
    #     x = self.fc2(x)
    #     x = torch.sigmoid(x)
    #     return x
    
cnn = CNN()
cnn = cnn.cuda()
optimizer = torch.optim.Adam(cnn.parameters(), lr= LR)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
maxacc = 0

def train(model, train_loader, optimizer, EPOCH):
    model.train()
    
    for i in range(EPOCH):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data.cuda())
            target = target.cuda()
            loss = F.binary_cross_entropy(output, target.float().reshape(20,1)) 
            loss.backward()
            optimizer.step()
        print(i,"   loss:",loss)
        if (i + 1) % 1 == 0:
            test(model, device, test_loader, i)
            
    



def test(model, device, test_loader, index):
    global maxacc
    model.eval()
    test_loss = 0
    correct = total = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            
            data, target = data.to(device),target.to(device).float().reshape(20, 1)
            output = model(data)
            test_loss += F.binary_cross_entropy(output, target, reduction='sum').item() # 将一批的损失相加
            target = target.cpu()
            # print(target)
            pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in output]).numpy()
            correct += float((pred == target.detach().numpy()).astype(int).sum())
            total += float(target.size(0))
    accuracy = correct / total
    if accuracy > maxacc:
        maxacc = accuracy
        torch.save(cnn.state_dict(), 'cnn.pth')
    print('Epoch: ', index, '| test accuracy: % .3f' %accuracy, '| max accuracy: % .3f' %maxacc, '|test-loss: %3f' %test_loss)

class_names = ['cat', 'dog']
model_state_dict  = torch.load('cnn.pth')
cnn.load_state_dict(model_state_dict)
cnn.eval()
#train(cnn, train_loader, optimizer, EPOCH)
image_PIL = Image.open('test1/test1/5.jpg').convert('RGB')
plt.imshow(image_PIL)
#
image_tensor = transform(image_PIL)
# 以下语句等效于 image_tensor = torch.unsqueeze(image_tensor, 0)
image_tensor.unsqueeze_(0)
# 没有这句话会报错
image_tensor = image_tensor.to(device)
 
out = cnn(image_tensor)
pred = torch.tensor([[1] if num[0] >= 0.5 else [0] for num in out]).to(device)
plt.title(class_names[pred])
