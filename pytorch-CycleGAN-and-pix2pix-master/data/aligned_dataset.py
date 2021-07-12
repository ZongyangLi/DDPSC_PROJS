import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)

        self.AB_paths = sorted(make_dataset(self.dir_AB))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5)),
                          ]

        self.transform = transforms.Compose(transform_list)
        
        transform_list_gray = [transforms.ColorJitter(0.8,0.8,0.3,0),  # transforms.ColorJitter(0.8,0.2,0.05,0)
                               transforms.Grayscale(3)]
        
        self.gray_transform = transforms.Compose(transform_list_gray)

    # get item source code
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
          
        
        # start color variation
        w = AB.size[0]/2
        AB_VARY = self.gray_transform(AB)
        AB_pix = np.array(AB)
        ABV_pix = np.array(AB_VARY)
        AB_pix[:,:w] = ABV_pix[:,:w] # gray to color
        #AB_pix[:,:w] = ABV_pix[:,:w] # color to color
        AB = Image.fromarray(AB_pix)
          
        # end color variation 
        
        
        '''
        # save pair image
        out_dir = '/media/zli/data/dcgan/terra_s2_rgb/shink_crop/gray_to_color'
        out_path = os.path.join(out_dir, os.path.basename(AB_path))
        save_image(AB_pix, out_path)
        '''
        AB = self.transform(AB)
        
  
        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
  
        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]
  
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
  
        return {'A': A, 'B': B,
                'A_paths': AB_path, 'B_paths': AB_path}


    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'
