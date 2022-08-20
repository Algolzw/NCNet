## [AIM 2022] Fast Nearest Convolution for Real-Time Image Super-Resolution

![ts](figs/ts.png)

### Dependencies
- OS: Ubuntu 18.04
- Python: Python 3.7
- Tensorflow 2.9.1
- nvidia :
   - cuda: 11.2
   - cudnn: 8.1.0
- Other reference requirements


### Model Training
```python3
python main.py
```
Then the trained keras model will be saved in ```ckpt/basenet/model``` folder.

### Model Validation
```python3
python eval.py
```
Then the results of original keras model will be saved in ```original_output``` folder and you can calculate the validation PSNR by run ```python calculate_PSNR.py```

### Convert to TFLite
``` bash
python generate_tflite.py
```
Then the converted tflite model will be saved in ```TFLite ``` folder.

### TFLite Model Validation
``` bash
python test_tflite.py
```
Then the results of TFLite model will be saved in ```results ``` folder.



### Other Details

* The input image range is [0, 255].
* Number of parameters: 52,279 (53K)
* Average PSNR on DIV2K validation data: 30.27 dB
* Training data: DIV2K.