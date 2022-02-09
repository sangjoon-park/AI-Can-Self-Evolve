import numpy as np
from PIL import Image
from torch.utils import data
import glob
import cv2
import utils

from torchvision import transforms as pth_transforms

import main_dino

parser = main_dino.get_args_parser()
args = parser.parse_args()

class CXR_Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_dir, transforms=None, mode='train', labeled=False):
        'Initialization'
        self.dim = (256, 256)
        self.n_classes = 1
        self.transforms = transforms
        self.mode = mode
        self.labeled = labeled

        if args.total_folds == 0:
            self.total_folds = ['labeled']
            self.pseudo_folds = None
        elif args.total_folds == 1:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0']
        elif args.total_folds == 2:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0', 'fold_1']
        elif args.total_folds == 3:
            self.total_folds = ['labeled']
            self.pseudo_folds = ['fold_0', 'fold_1', 'fold_2']

        self.test_fold = ['test']

        self.total_images = {}

        # PNG and JPG image lists
        if self.mode == 'train':
            if self.labeled == True:
                for fold in self.total_folds:   # Fix total folds to 1 during self-training
                    png_lists = glob.glob(data_dir + '{}/'.format(fold) + '**/*.png', recursive=True)
                    jpg_lists = glob.glob(data_dir + '{}/'.format(fold) + '**/*.jpg', recursive=True)
                    for png in png_lists:
                        self.total_images[png] = 'label'
                    for jpg in jpg_lists:
                        self.total_images[jpg] = 'label'

            if self.labeled == False:
                for p_fold in self.pseudo_folds:
                    p_png_lists = glob.glob(data_dir + '{}/'.format(p_fold) + '**/*.png', recursive=True)
                    p_jpg_lists = glob.glob(data_dir + '{}/'.format(p_fold) + '**/*.jpg', recursive=True)
                    for p_png in p_png_lists:
                        self.total_images[p_png] = 'pseudo'
                    for p_jpg in p_jpg_lists:
                        self.total_images[p_jpg] = 'pseudo'

        elif self.mode == 'test':
            for t_fold in self.test_fold:
                png_lists = glob.glob(data_dir + '{}/'.format(t_fold) + '**/*.png', recursive=True)
                jpg_lists = glob.glob(data_dir + '{}/'.format(t_fold) + '**/*.jpg', recursive=True)
                for png in png_lists:
                    self.total_images[png] = 'label'
                for jpg in jpg_lists:
                    self.total_images[jpg] = 'label'

        self.total_images_list = sorted(self.total_images.keys())
        self.selected_images = self.total_images_list

        print('A total of %d image data were generated.' % len(self.selected_images))
        self.n_data = len(self.selected_images)

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_data

    def __getitem__(self, index):
        'Generates one sample of data'

        img_path = self.total_images_list[index]
        image = cv2.imread(img_path, 1)

        image = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_LINEAR)

        image = Image.fromarray(image)

        # Apply DINO augmentation
        if not self.transforms == None:
            images = self.transforms(image)

        # Apply NO augmentation
        elif self.transforms == None:
            image = pth_transforms.Compose(
                [utils.GaussianBlurInference(),
                 pth_transforms.ToTensor()])(image)

        idx = img_path

        # Make label
        if self.labeled == True:
            if 'Normal' in idx:
                label = 0
            elif 'Tuberculosis' in idx:
                label = 1
        else:
            # Label without meaning
            label = 9999

        if self.mode == 'train':
            return images, label
        else:
            return image, label