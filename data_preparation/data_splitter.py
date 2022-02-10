import numpy as np
import os
import glob
import cv2
from tqdm import tqdm
import random
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('data_splitter', add_help=False)
    parser.add_argument("--train_folder", default='PATH/TO/TRAIN/', type=str, help='Folder containing all train data')
    parser.add_argument("--test_folder", default='PATH/TO/TEST/', type=str, help='Folder containing all test data')
    parser.add_argument("--save_dir", default='PATH/TO/SAVE/', type=str, help='Path to your dcm files')

    return parser

parser = argparse.ArgumentParser('data_splitter', parents=[get_args_parser()])
args = parser.parse_args()

# RANDOM SEED
seed = 0

n_fold = 3

imgs = []
imgs.extend(glob.glob(args.train_folder + '**/*.png', recursive=True))
imgs.extend(glob.glob(args.train_folder + '**/*.jpg', recursive=True))
imgs.extend(glob.glob(args.train_folder + '**/*.jpeg', recursive=True))

random.seed(seed)
random.shuffle(imgs)

# SPLIT SMALL-SIZED LABELED DATA (10%)
label_fold = len(imgs) // 10

labeled_imgs = imgs[:label_fold]
remained_imgs = imgs[label_fold:]

for one_img in tqdm(labeled_imgs):
    if 'Normal' in one_img and not 'Tuberculosis' in one_img:
        label = 'Normal'
    elif 'Tuberculosis' in one_img and not 'Normal' in one_img:
        label = 'Tuberculosis'
    else:
        raise NameError('Not a valid label.')

    save_path = args.save_dir + 'labeled/' + label + '/'
    os.makedirs(save_path, exist_ok=True)

    # SAVE IMAGE AFTER NORMALIZATION
    img = cv2.imread(one_img, 0)
    img = ((img - img.min()) / (img.max() - img.min())) * 255.
    img = img.astype('uint8')

    cv2.imwrite(save_path + one_img.split('/')[-1], img)

remain_length_fold = len(remained_imgs) // n_fold

# SPLIT UNLABELED 3 FOLDS DATA (90%)
for fold in range(n_fold):
    if not (fold+1) == n_fold:
        fold_imgs = remained_imgs[fold * remain_length_fold:(fold+1) * remain_length_fold]
    elif fold+1 == n_fold:
        fold_imgs = remained_imgs[fold * remain_length_fold:]

    for one_img in tqdm(fold_imgs):
        if 'Normal' in one_img and not 'Tuberculosis' in one_img:
            label = 'Normal'
        elif 'Tuberculosis' in one_img and not 'Normal' in one_img:
            label = 'Tuberculosis'
        else:
            raise NameError('Not a valid label.')

        save_path = args.save_dir + 'fold_{}/'.format(fold) + label + '/'
        os.makedirs(save_path, exist_ok=True)

        # Normalize 후 Save
        img = cv2.imread(one_img, 0)
        img = ((img - img.min()) / (img.max() - img.min())) * 255.
        img = img.astype('uint8')

        cv2.imwrite(save_path + one_img.split('/')[-1], img)


# TEST DATA
test_imgs = []
test_imgs.extend(glob.glob(args.test_folder + '**/*.png', recursive=True))
test_imgs.extend(glob.glob(args.test_folder + '**/*.jpg', recursive=True))
test_imgs.extend(glob.glob(args.test_folder + '**/*.jpeg', recursive=True))

for one_img in tqdm(test_imgs):
    if 'Normal' in one_img and not 'Tuberculosis' in one_img:
        label = 'Normal'
    elif 'Tuberculosis' in one_img and not 'Normal' in one_img:
        label = 'Tuberculosis'
    else:
        raise NameError('Not a valid label.')

    save_path = args.save_dir + 'test/' + label + '/'
    os.makedirs(save_path, exist_ok=True)

    # SAVE IMAGE AFTER NORMALIZATION
    img = cv2.imread(one_img, 0)
    img = ((img - img.min()) / (img.max() - img.min())) * 255.
    img = img.astype('uint8')

    cv2.imwrite(save_path + one_img.split('/')[-1], img)