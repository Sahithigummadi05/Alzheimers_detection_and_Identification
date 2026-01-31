import os
import numpy as np
import cv2
import joblib
from nilearn import datasets
from nilearn.image import smooth_img
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data_dir = os.path.join(os.path.expanduser('~'), 'nilearn_data')
IMG_SIZE = 50
N_SUBJECTS = 100 

def fetch_and_process_data():
    oasis_dataset = datasets.fetch_oasis_vbm(n_subjects=N_SUBJECTS, data_dir=data_dir)
    gray_matter_maps = oasis_dataset.gray_matter_maps
    cdr = oasis_dataset.ext_vars['cdr'].astype(float)
    
    y_labels = []
    for val in cdr:
        if np.isnan(val) or val > 0.0:
            y_labels.append(1)
        else:
            y_labels.append(0)
    
    y_labels = np.array(y_labels)
    processed_data = []
    
    for i, img_path in enumerate(gray_matter_maps):
        try:
            img_nii = smooth_img(img_path, fwhm=1)
            img_3d = img_nii.get_fdata()
            mid_slice = img_3d.shape[2] // 2
            slice_2d = img_3d[:, :, mid_slice]
            resized = cv2.resize(slice_2d, (IMG_SIZE, IMG_SIZE))
            processed_data.append(resized.flatten())
        except Exception:
            continue

    return np.array(processed_data), y_labels

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    return clf

if __name__ == "__main__":
    X, y = fetch_and_process_data()
    model = train_model(X, y)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    joblib.dump(model, './models/alzheimer_model.pkl')
    print("SUCCESS")