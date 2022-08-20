import os
from div2k import DIV2K
from model.basenet import basenet
from train import BaseNetTrainer
import tensorflow as tf

model_name = 'basenet'
downgrade = 'bicubic'
scale = 3

n_feat = 32
batch_size = 64

steps = 500000
evaluate_every = 1000


if __name__ == '__main__':

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    df2k_train = DIV2K(scale=scale, subset='train')
    df2k_valid = DIV2K(scale=scale, subset='valid')

    train_ds = df2k_train.dataset(batch_size=batch_size, random_transform=True)
    valid_ds = df2k_valid.dataset(batch_size=1, random_transform=False, repeat_count=1)
    print(f'train dataset: {len(df2k_train)}, valid dataset: {len(df2k_valid)}')

    model = basenet(n_feat=n_feat, out_c=3, scale_factor=scale)
    trainer = BaseNetTrainer(
        model=model,
        checkpoint_dir=f'ckpt/{model_name}')

    trainer.train(train_ds, valid_ds,
            steps=steps,
            evaluate_every=evaluate_every,
            save_best_only=True)
