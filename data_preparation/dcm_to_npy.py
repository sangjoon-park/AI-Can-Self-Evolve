import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import SimpleITK as sitk
import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('dicom_to_npy', add_help=False)
    parser.add_argument("--dir", default='PATH/TO/DCM/', type=str, help='Path to your dcm files')
    parser.add_argument("--save_dir", default='PATH/TO/SAVE/', type=str, help='Path to your dcm files')

    return parser

parser = argparse.ArgumentParser('dicom_to_npy', parents=[get_args_parser()])
args = parser.parse_args()

dcms = glob.glob(args.dir + '**/*.dcm', recursive=True)
dcms.extend(glob.glob(args.dir + '**/*.DCM', recursive=True))

print('>>> Total of {} DCMs are detected.'.format(len(dcms)))

for dcm in tqdm(dcms):
    dicom = pydicom.dcmread(dcm)
    img = dicom.pixel_array
    monochrome = dicom.PhotometricInterpretation

    # TO PREVENT ERRORS IN NIH DATA
    if monochrome == 'YBR_FULL_422':
        continue

    img = cv2.resize(img, dsize=(1024, 1024))
    img = (img - img.min()) / (img.max() - img.min())

    img = img * 255.
    img = img.astype(np.uint8)

    path = dcm.split('/')[-1].split('.dcm')[0]
    path = path.replace('/', '_') + '____' + monochrome
    save_path = args.save_dir + path + '.png'

    save_path_dir_list = save_path.split('/')[:-1]
    str = ''
    for name in save_path_dir_list:
        str = str + name + '/'

    os.makedirs(str, exist_ok=True)
    cv2.imwrite(save_path, img)