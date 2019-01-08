import os
import collections
import torch
import vision.torchvision.transforms as transforms
import numpy as np
import scipy.misc as m
import random
from torch.utils import data
from PIL import Image
from torch import nn

def parseTxt(filepath):
    with open(filepath, 'r') as f:
        data = f.readlines()
    return data


class culaneLoader(data.Dataset):
    def __init__(self, root='/data6/public_datasets/traffic/CULane', split="train", is_transform=False, img_size=None):
        self.root = root
        self.split = split
        self.img_size = [192, 320]
        self.is_transform = is_transform
        self.n_classes = 5
        self.files = collections.defaultdict(list)
        self.trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.25, hue=0.1),
        ])
        self.row_end = 1000
        self.phase = split

        for split in ["train", "val"]:
            self.files[split] = parseTxt(root+'/list/'+split+'.txt')#[:5000]
            if split=='val':
                self.files[split] = parseTxt(root+'/list/'+split+'.txt')[:100]

            print("{} number: {:d}".format(split, len(self.files[split])))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_path = self.files[self.split][index][0:-1]
        lbl_path = '/laneseg_label_w16' + img_path.replace('.jpg', '.png', 1)
        img_path = self.root + img_path
        lbl_path = self.root + lbl_path

        # Load from CULane small
        img_path = img_path.replace('/data6/public_datasets/traffic/CULane/', '/data1/ymli/datasets/CULane_small/')
        lbl_path = lbl_path.replace('/data6/public_datasets/traffic/CULane/', '/data1/ymli/datasets/CULane_small/')

        img = m.imread(img_path)#[295:,:,:]
        lbl = m.imread(lbl_path)#[295:,:]

        # Save images
        # img_path = img_path.replace('/data6/public_datasets/traffic/CULane/', '/data1/ymli/datasets/CULane_small/')
        # lbl_path = lbl_path.replace('/data6/public_datasets/traffic/CULane/', '/data1/ymli/datasets/CULane_small/')
        # img = m.imresize(img, self.img_size, interp='nearest')
        # lbl = m.imresize(lbl, self.img_size, interp='nearest')
        # lbl_new = np.zeros((self.img_size)).astype('uint8')
        # trans_list = [3,1,2,4]
        # for i in range(1, 5):
        #     lbl_new[lbl==i] = trans_list[i-1]
        # imgdir, _ = os.path.split(img_path)
        # lbldir, _ = os.path.split(lbl_path)
        # if not os.path.exists(imgdir):
        #     os.makedirs(imgdir)
        # if not os.path.exists(lbldir):
        #     os.makedirs(lbldir)
        # m.imsave(img_path, img)
        # m.imsave(lbl_path, lbl_new)


        if self.phase == 'train' and self.is_transform:
            img = self.trans(Image.fromarray(img))
            img = np.asarray(img)

        img = transforms.ToTensor()(img)    
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def decode_segmap(self, temp):
      
        color_map = [[0,0,0],        
                     [0,0,255],      
                     [0,255,0],      
                     [255,0,0], 
                     [255,255,0], 
                     [255,0,255], 
                     [0,255,255],      
                     [255,0,128], 
                     [128,0,0], 
                     [128,128,0], 
                    ]

        mask_color = np.zeros((temp.shape[0], temp.shape[1], 3))
        for l in range(0, self.n_classes):
          mask_color[temp==l,:] = color_map[l]

        return mask_color

if __name__ == '__main__':
    import visdom
    vis = visdom.Visdom(server='http://localhost', port=8097, env='display')
    img_rows, img_cols = 288, 800
    win_image = vis.image(np.ndarray((3, img_rows, img_cols)), opts=dict(title='Image'))
    win_mask = vis.image(np.ndarray((3, img_rows, img_cols)), opts=dict(title='Mask'))

    local_path = '/data6/public_datasets/traffic/CULane'
    dst = culaneLoader(local_path, split="train", is_transform=False)
    trainloader = data.DataLoader(dst, batch_size=5, shuffle=False, num_workers=4)

    for i, data in enumerate(trainloader):
        imgs, labels = data
        imgs, labels = imgs[0], labels[0]
        labels = dst.decode_segmap(labels.numpy())

        net = nn.Sequential(
                    nn.Conv2d(1, 4, 3),
                    nn.ReLU())

        pix_embedding = net(input)
        image_shape = (pix_embedding.get_shape().as_list()[1], pix_embedding.get_shape().as_list()[2])

     #   disc_loss, l_var, l_dist, l_reg =  lanenet_discriminative_loss.discriminative_loss(
     #       pix_embedding, labels, 4, image_shape, 0.5, 3.0, 1.0, 1.0, 0.001)


        # Show results
        vis.image(imgs.numpy(), opts=dict(title='Image'), win=win_image)
        vis.image(labels.transpose(2, 0, 1), opts=dict(title='Mask'), win=win_mask)
        print(i)
        
        # wait = input('PRESS KEY')
        # print('')
