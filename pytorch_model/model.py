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

workdir = '/home/qianwen/KDD/cfree/ckpt/alexnet_50'
if not os.path.exists(workdir):
    os.mkdir(workdir)

dataset_name = 'AmazonFashion6ImgPartitioned.npy'
dataset_dir = '/home/qianwen/KDD/dataset/amazon/'
dataset = np.load(dataset_dir + dataset_name, encoding = 'bytes')

[user_train, user_validation, user_test, Item, usernum, itemnum] = dataset
non_cold_usernum = 20391

def preprocess(img_s, mean=np.array([0.43, 0.47, 0.49]), std=np.array([1.0, 1.0, 1.0])):
    img_a = np.asarray(Image.open(BytesIO(img_s)))
    if len(img_a.shape) != 3 or img_a.shape[2] != 3:
        return None
    else:
        return (imresize(img_a, (224, 224, 3)) / 255 - np.array(mean)) / np.array(std)

latent_d = 50
batch_size = 128
net = torchvision.models.AlexNet(num_classes=latent_d)
# for param in net.parameters():
#    param.requires_grad = False
# net.fc = torch.nn.Linear(512, latent_d)

class Siamese(torch.nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.net = net
        self.kernel = torch.nn.Parameter(torch.randn(usernum, latent_d).cuda())
        
    def forward(self, input_i, input_j, input_index):
        output_i = self.net(input_i)
        output_j = self.net(input_j)
        u = torch.cat([self.kernel.select(0, k) for k in input_index]).resize(len(input_index), latent_d)
        output_i = output_i * u
        output_j = output_j * u
        return output_i, output_j

class DataGenerator():
    def __init__(self, source, batch_size, dim, mean=np.array([0.43, 0.47, 0.49])):
        self.source = source
        self.batch_size = batch_size
        self.mean = mean
        # self.std = std
        self.dim = dim

    def batches(self, mode='sampling'):
        u = 0
        while u < len(self.source):
            img_i = np.empty((self.batch_size, *self.dim))
            img_j = np.empty((self.batch_size, *self.dim))
            index = []
            k = 0
            while k < self.batch_size and u < len(self.source):
                if mode == 'sampling':
                    u = random.randint(0, len(self.source)-1)
                i = self.source[u][random.randint(0, len(self.source[u])-1)][b'productid']
                img_item_i = preprocess(Item[i][b'imgs'])
                u_imgs = list(set([e[b'productid'] for e in user_train[u]] + 
                                  [e[b'productid'] for e in user_validation[u]] + 
                                  [e[b'productid'] for e in user_test[u]]))
                if mode == 'traversal':
                    u += 1
                if len(u_imgs) < 7 or img_item_i is None:
                    continue
                j = random.randint(0, len(Item)-1)
                img_item_j = preprocess(Item[j][b'imgs'])
                while j in u_imgs or img_item_j is None:
                    j = random.randint(0, len(Item)-1)
                    img_item_j = preprocess(Item[j][b'imgs'])
                img_j[k] = img_item_j
                img_i[k] = img_item_i
                index.append(u)
                k += 1
            if k == self.batch_size:
                yield img_i, img_j, index

def evaluate(model, mode="sampling", batch_num=10):
    scores = []
    validation_data = DataGenerator(source=user_validation, batch_size=128, dim=(224, 224, 3))
    i = 0
    for batch in validation_data.batches(mode=mode):
        if mode == "sampling" and i > batch_num:
            break
        i += 1
        img_i, img_j, index = batch
        input_i = V(torch.from_numpy(img_i).float().permute(0,3,1,2)).cuda()
        input_j = V(torch.from_numpy(img_j).float().permute(0,3,1,2)).cuda()
        output_i, output_j = model(input_i, input_j, index)
        scores += (torch.sum(output_i, 1) > torch.sum(output_j, 1)).data.tolist()
    return round(sum(scores) / len(scores), 3)

siamese_net = Siamese()
siamese_net.cuda()
# siamese_net.load_state_dict(torch.load('/home/qianwen/KDD/cfree/ckpt/alexnet_50/iteration_1'))
training_data = DataGenerator(source=user_train, batch_size=batch_size, dim=(224, 224, 3))
learning_rate = 1e-4
params = []
for name, param in siamese_net.named_parameters():
    if param.requires_grad:
        params.append({'params': [param], 
			'weight_decay': 0.1 if name=='kernel' else 1e-3,
			'lr': 1e-4 if name=='kernel' else 1e-4})
optimizer = torch.optim.Adam(params)
f = open(os.path.join(workdir, 'log.txt'), 'w')
iteration = 0
step = 0
loss_buffer = []
one_iteration_steps = int(sum([len(v) for k,v in user_train.items()]) / batch_size / 10)  
print("Training begin:")
for batch in training_data.batches():
    if step >= one_iteration_steps:
        score = evaluate(siamese_net, mode='traversal')
        print("Iteration {} Validation Score {}".format(iteration, score))
        f.write("Iteration {} Validation Score {}".format(iteration, score))
        f.flush()
        torch.save(siamese_net.state_dict(), os.path.join(workdir, 'iteration_{}'.format(iteration)))
        iteration += 1
        step = 0
    img_i, img_j, index = batch
    input_i = V(torch.from_numpy(img_i).float().permute(0,3,1,2)).cuda()
    input_j = V(torch.from_numpy(img_j).float().permute(0,3,1,2)).cuda()
    output_i, output_j = siamese_net(input_i, input_j, index)
    # L1 distance with Sigmoid
    distance = torch.sum(torch.nn.LogSigmoid()(torch.sum(output_i - output_j, 1)))
    loss = -distance
    loss_buffer.append(loss)
    avg_loss = sum(loss_buffer[-20:]) / min(20, len(loss_buffer))
    print('Iteration {}, Step {}, Loss {}'.format(iteration, step, avg_loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1
f.close()
