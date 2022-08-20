import os
from model.basenet import basenet
from model import resolve
import tensorflow as tf
import cv2
from tqdm import tqdm
import numpy as np


model_name = 'basenet'
downgrade = 'bicubic'
scale = 3
image_folder = "/data/dataset/aim22/aimsr/DIV2K_valid_LR_bicubic/X3/"
output_folder = "original_output"

if __name__ == '__main__':

    os.makedirs(output_folder, exist_ok=True)
    model_dir = f".ckpt/{model_name}/model"
    model = tf.keras.models.load_model(model_dir)
    model.summary()

    image_list = os.listdir(image_folder)
    psnr_values = []
    for image_name in tqdm(image_list):
        img = cv2.imread(image_folder + image_name)
        lr = np.expand_dims(img, axis=0)
        lr = tf.convert_to_tensor(lr)
        sr = resolve(model, lr)[0]

        sr = sr.numpy().round()
        cv2.imwrite(f'{output_folder}/{image_name}', sr)
        

