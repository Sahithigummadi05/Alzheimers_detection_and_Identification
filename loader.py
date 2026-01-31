import os
import numpy as np
import cv2
from nilearn import datasets
from nilearn.image import smooth_img

def load_and_preprocess_oasis(n_subjects=150, img_size=128):
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    gray_matter_maps = oasis_dataset.gray_matter_maps
    cdr = oasis_dataset.ext_vars['cdr'].astype(float)
    
    raw_labels = np.nan_to_num(cdr, nan=0.0)
    y = []
    for val in raw_labels:
        if val == 0.0: y.append(0)
        elif val == 0.5: y.append(1)
        elif val == 1.0: y.append(2)
        else: y.append(3)
    y = np.array(y)
    
    X = []
    for img_path in gray_matter_maps:
        img_nii = smooth_img(img_path, fwhm=1)
        img_3d = img_nii.get_fdata()
        mid_slice = img_3d.shape[2] // 2
        slice_2d = img_3d[:, :, mid_slice]
        
        resized = cv2.resize(slice_2d, (img_size, img_size))
        normalized = (resized - np.min(resized)) / (np.max(resized) - np.min(resized))
        rgb_slice = np.stack([normalized]*3, axis=-1)
        X.append(rgb_slice)
        
    return np.array(X), y