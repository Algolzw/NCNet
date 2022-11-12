# Copyright 2021 by Andrey Ignatov. All Rights Reserved.

# The following instructions will show you how to test your converted (quantized / floating-point) TFLite model
# on the real images

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
import os
import cv2

from tensorflow.lite.python import interpreter as interpreter_wrapper
from tqdm import tqdm
import time

if __name__ == "__main__":

    # Specify the name of your TFLite model and the location of the sample test images
    image_folder = "/data/dataset/aim22/aimsr/DIV2K_valid_LR_bicubic/X3/"
    model_file = "TFLite/model_none.tflite"
    output_folder = "results"

    # Load your TFLite model
    interpreter = interpreter_wrapper.Interpreter(model_path=model_file, num_threads=32)

    input_details = interpreter.get_input_details()
    print(input_details)

    output_details = interpreter.get_output_details()
    print(output_details)

    # Process test images and save the results
    image_list = os.listdir(image_folder)
    image_list.sort()

    os.makedirs(output_folder, exist_ok=True)

    for image_name in tqdm(image_list):

        img = Image.open(image_folder + image_name)
        input_data = np.expand_dims(img, axis=0)
        input_data = input_data.astype(input_details[0]['dtype'])

        interpreter.resize_tensor_input(input_details[0]['index'], input_data.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        results = interpreter.get_tensor(output_details[0]['index'])
        results = np.clip(results, 0, 255).squeeze()

        img = Image.fromarray(results.astype(np.uint8), mode="RGB")
        img.save(f"{output_folder}/" + image_name)


