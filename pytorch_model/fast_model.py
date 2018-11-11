from io import StringIO, BytesIO
from PIL import Image
from scipy.misc import imresize
import random
import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import zoom
import os

import torch
import torchvision
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

workdir = '/home/qianwen/KDD/cfree/ckpt/alexnet_pretrained_50_dropout'
if not os.path.exists(workdir):
    os.mkdir(workdir)

dataset_name = 'AmazonFashion6ImgPartitioned.npy'
dataset_dir = '/home/qianwen/KDD/dataset/amazon/'
dataset = np.load(dataset_dir + dataset_name, encoding = 'bytes')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
non_cold_usernum = 20391

latent_d = 50
batch_size = 128
net = torchvision.models.alexnet(pretrained=True)
# for param in net.parameters():
#    param.requires_grad = False
# net.fc = torch.nn.Linear(512, latent_d)

class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.features = net.features
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.5), net.classifier[1], net.classifier[2], torch.nn.Dropout(0.5), net.classifier[4])
        self.linear = torch.nn.Linear(4096, latent_d)
        self.kernel = torch.nn.Parameter(torch.randn(usernum, latent_d).cuda())
        
    def forward(self, input_i, input_j, input_index):
        output_i = self.classifier((self.features(input_i).resize(batch_size, 256*6*6)))
        output_i = self.linear(output_i)
        output_j = self.classifier((self.features(input_j).resize(batch_size, 256*6*6)))
        output_j = self.linear(output_j)
        u = torch.cat([self.kernel.select(0, k) for k in input_index]).resize(batch_size, latent_d)
        output_i = output_i * u
        output_j = output_j * u
        return output_i, output_j

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
       ])

class AmazonDataset(Dataset):    
    def __init__(self, source, transform):
        self.source = source
        self.transform = data_transform
        
    def __len__(self):
        return sum([len(v) for k,v in self.source.items()])
        
    def __getitem__(self, idx):   
        u = idx%usernum
        i = self.source[u][random.randint(0, len(self.source[u])-1)][b'productid']
        img_i = self.transform(Image.open(BytesIO(Item[i][b'imgs'])))
        u_imgs = list(set([e[b'productid'] for e in user_train[u]] +
                          [e[b'productid'] for e in user_validation[u]] +
                          [e[b'productid'] for e in user_test[u]]))
        if img_i.shape[0] == 1:
            img_i = img_i.repeat(3,1,1)
        j = i
        while j in u_imgs:
            j = random.randint(0, itemnum-1)
        img_j = self.transform(Image.open(BytesIO(Item[j][b'imgs'])))
        if img_j.shape[0] == 1:
            img_j = img_j.repeat(3,1,1)
        assert(img_i.shape[0] == 3 and img_j.shape[0] == 3)
        return u, img_i, img_j


def evaluate(model, mode="sampling", batch_num=10):
    scores = []
    validation_dataset = AmazonDataset(user_validation, transforms)
    validation_generator = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=32, drop_last=True)
    model.eval()
    for index, input_i, input_j in validation_generator:
        input_i = input_i.cuda()
        input_j = input_j.cuda()
        output_i, output_j = model(input_i, input_j, index)
        scores += (torch.sum(output_i, 1) > torch.sum(output_j, 1)).data.tolist()
    model.train()
    return sum(scores) / len(scores)

siamese_net = Siamese()
siamese_net.cuda()
# siamese_net.load_state_dict(torch.load('/home/qianwen/KDD/cfree/ckpt/alexnet_50/iteration_1'))
training_dataset = AmazonDataset(user_train, transforms)
training_generator = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=16, drop_last = True)
learning_rate = 1e-4
params = []
for name, param in siamese_net.named_parameters():
    if param.requires_grad:
        params.append({'params': [param], 
			'weight_decay': 1e-2 if name=='kernel' else 1e-3,
			'lr': 1e-2 if name=='kernel' else 1e-4})
optimizer1 = torch.optim.Adam(params)
params = []
for name, param in siamese_net.named_parameters():
    if param.requires_grad:
        params.append({'params': [param],
                        'weight_decay': 1e-2 if name=='kernel' else 1e-3,
                        'lr': 1e-4 if name=='kernel' else 1e-4})
optimizer2 = torch.optim.Adam(params)
f = open(os.path.join(workdir, 'log.txt'), 'w')
max_iteration = 50
print("Training begin:")
for iteration in range(max_iteration):
    step = 0
    score = evaluate(siamese_net, mode='traversal')
    print("Iteration {} Validation Score {}".format(iteration, score))
    f.write("Iteration {} Validation Score {}\n".format(iteration, score))
    f.flush()
    torch.save(siamese_net.state_dict(), os.path.join(workdir, 'iteration_{}'.format(iteration)))
    for index, input_i, input_j in training_generator:
        input_i = input_i.cuda()
        input_j = input_j.cuda()
        output_i, output_j = siamese_net(input_i, input_j, index)
        distance = torch.sum(torch.nn.LogSigmoid()(torch.sum(output_i - output_j, 1)))
        score = round(float(torch.sum(torch.sum(output_i, 1) > torch.sum(output_j, 1))) / batch_size, 3)
        loss = -distance
        print('Iteration {}, Step {}, Loss {}, Score {}'.format(iteration, step, loss, score))
        if iteration < 4:
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
        else:
            optimizer2.zero_grad()
            loss.backward()
            optimizer2.step()
        step += 1
    
f.close()
