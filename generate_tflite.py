import tensorflow as tf
import numpy as np
import os
import cv2

from tqdm import tqdm




class DIV2K(tf.keras.utils.Sequence):
    def __init__(self, data_root, scale_factor=3, batch_size=32, patch_size=192, type='train'):
        self.data_root = data_root
        self.scale_factor = scale_factor
        self.batch_size = batch_size
        self.image_ids = range(1, 801)
        self.patch_size = patch_size
        self.type = 'train'

    def __getitem__(self, idx):
        if self.patch_size > 0:
            start = idx * self.batch_size
            end = (idx + 1) * self.batch_size
            batch_ids = self.image_ids[start:end]
            lr_batch = np.zeros((self.batch_size, self.patch_size // self.scale_factor, self.patch_size // self.scale_factor, 3), dtype='float32')
            hr_batch = np.zeros((self.batch_size, self.patch_size, self.patch_size, 3), dtype='float32')
            for i, id in enumerate(batch_ids):
                lr, hr = self._get_image_pair(id)
                lr = np.expand_dims(lr, 0)
                lr = np.expand_dims(lr, -1)
                hr = np.expand_dims(hr, 0)
                hr = np.expand_dims(hr, -1)
                lr_batch[i] = lr
                hr_batch[i] = hr
            return lr_batch, hr_batch
        else:
            # Return 1 image pair if not returning an image patch
            return self._get_image_pair(self.image_ids[idx])

    def __len__(self):
        if self.patch_size > 0:
            length = int(np.ceil(len(self.image_ids) / self.batch_size))
        else:
            length = len(self.image_ids)
        return length

    def _get_image_pair(self, id):
        image_id = f'{id:04}'
        hr_file = f'{self.data_root}/DIV2K_{self.type}_HR/{image_id}.png'
        hr_y = self._get_y_chan(hr_file).astype(np.float32)
        if self.patch_size > 0:
            hr_y = self._crop_center(hr_y, self.patch_size, self.patch_size)

        lr_file = f'{self.data_root}/DIV2K_{self.type}_LR_bicubic/X{self.scale_factor}/{image_id}x{self.scale_factor}.png'
        lr_y = self._get_y_chan(lr_file).astype(np.float32)
        if self.patch_size > 0:
            lr_y = self._crop_center(lr_y, self.patch_size // self.scale_factor, self.patch_size // self.scale_factor)
        return lr_y, hr_y

    def _get_y_chan(self, image_path):
        im = cv2.imread(image_path)
        return im

    def _crop_center(self, im, crop_h, crop_w):
        startx = im.shape[1] // 2 - (crop_w // 2)
        starty = im.shape[0] // 2 - (crop_h // 2)
        return im[starty:starty + crop_h, startx:startx + crop_w]

def representative_dataset_gen_model():
    div2k = DIV2K('/data/dataset/aim22/aimsr', patch_size=-1)
    input_shape = [1, 360, 640, 3]
    for i in tqdm(range(1)):
        x, _ = div2k[i]
        if x.shape[0] > input_shape[1] and x.shape[1] > input_shape[2]:
            # crop to input shape starting for top left corner of image
            x = x[:input_shape[1], :input_shape[2]].astype(np.float32)
            x = np.expand_dims(x, 0)
            yield [x]
        else:
            continue

def representative_dataset_gen_model_none():
    div2k = DIV2K('/data/dataset/aim22/aimsr', patch_size=-1)
    for i in tqdm(range(100)):
        x, _ = div2k[i]
        x = x.astype(np.float32)
        x = np.expand_dims(x, 0)
        yield [x]


def generate_tflite(name, model_path, folder):
    if name == 'model':
        input_shape = [1, 360, 640, 3]
        rep_data = representative_dataset_gen_model
    else:
        input_shape = [1, None, None, 3]
        rep_data = representative_dataset_gen_model_none

    model = tf.saved_model.load(model_path)

    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape(input_shape)
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data

    if name == 'model' or name == 'model_none':
        print("====== uint8 quantification ======")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    tflite_model_path = f"{output_folder}/{name}.tflite"
    open(tflite_model_path, "wb").write(tflite_model)

    print(f'......{tflite_model_path}......')


if __name__ == "__main__":

    name1 = 'model'
    name2 = 'model_none'
    name3 = 'model_none_float'
    output_folder = 'TFLite'
    model_path = 'ckpt/basenet/model'

    os.makedirs(output_folder, exist_ok=True)
    
    generate_tflite(name1, model_path, model_path)
    generate_tflite(name2, model_path, model_path)
    generate_tflite(name3, model_path, model_path)






















