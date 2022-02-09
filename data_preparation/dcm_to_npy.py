import cv2
import pydicom
import numpy as np
import os
import glob
from tqdm import tqdm
import SimpleITK as sitk

# CONFIG
dir = 'PATH TO FOLDER CONTAINING .dcm FILES'
save_dir = 'PATH TO SAVE PROCESSED FILES'

# DATA type: NIH or others (preprocessing code varies depending on dataset sources)
data = 'NIH'

dcms = glob.glob(dir + '**/*.dcm', recursive=True)

print('>>> Total of {} DCMs are detected.'.format(len(dcms)))

if data == 'NIH':
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
        save_path = save_dir + path + '.png'

        save_path_dir_list = save_path.split('/')[:-1]
        str = ''
        for name in save_path_dir_list:
            str = str + name + '/'

        os.makedirs(str, exist_ok=True)
        cv2.imwrite(save_path, img)

else:
    ds = sitk.ImageFileReader()

    for dcm in tqdm(dcms):
        ds.SetFileName(dcm)
        img = sitk.GetArrayFromImage(sitk.ReadImage(dcm))[0]
        img = np.asarray(img)

        img = cv2.resize(img, dsize=(1024, 1024))
        img = (img - img.min()) / (img.max() - img.min())

        img = img * 255.
        img = img.astype(np.uint8)

        path = dcm.split('/')[-1].split('.dcm')[0]
        save_path = save_dir + path + '.png'
        save_path_dir_list = save_path.split('/')[:-1]
        str = ''
        for name in save_path_dir_list:
            str = str + name + '/'

        os.makedirs(str, exist_ok=True)
        cv2.imwrite(save_path, img)