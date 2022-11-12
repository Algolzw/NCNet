import numpy as np
import cv2
import torch

from ncnet import NCNet


def main():

    image_tensor = torch.tensor(
        cv2.imread('lr_0803x3.png')
    ).permute(2, 0, 1).unsqueeze(0).float()

    model = NCNet()

    with torch.no_grad():
        out_tensor = model(image_tensor)
    
    out_numpy = out_tensor[0].permute(1, 2, 0).numpy()
    cv2.imwrite('out.png', out_numpy)


if __name__ == '__main__':
    main()