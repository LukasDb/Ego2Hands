from tqdm import tqdm
import tensorflow as tf
import cv2
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import Config
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime



def get_preprocessed_datasets(config, num):
    x_train = []
    y_train = []
    x_eval = []
    y_eval = []
    print("Loading training images...")
    for idx in tqdm(range(num)):
        filename_img = os.path.join(config.dataset_train_preprocessed, "img", f"img_{idx:05}.png")
        filename_mask = os.path.join(config.dataset_train_preprocessed, "label", f"mask_{idx:05}.png")
        x_train.append(cv2.imread(filename_img))
        y_train.append(cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE))

    print("Loading evaluation images...")
    for idx in tqdm(range(2000)):
        filename_img = os.path.join(config.dataset_eval_preprocessed, "img", f"img_{idx:05}.png")
        filename_mask = os.path.join(config.dataset_eval_preprocessed, "label", f"mask_{idx:05}.png")
        x_eval.append(cv2.imread(filename_img))
        y_eval.append(cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE))
    
    return x_train, y_train, x_eval, y_eval


# ------------------------- DATASET AND CONSTANTS -------------------------------------

def main():
    CONTINUE_AT_IDX = 30000
    UNTIL_IDX = 3000000000
    PREPROCESS_EVAL = False

    config_path = os.path.join("configs", "config_tf.yml")
    config = Config(config_path)
    INPUT_SHAPE = config.input_shape

    train_loader = Ego2HandsData(config, mode = "train_seg")

    UNTIL_IDX = min(train_loader.__len__()-1, UNTIL_IDX) # limit 

    ensure_dir(os.path.join(config.dataset_train_preprocessed, "img"))
    ensure_dir(os.path.join(config.dataset_train_preprocessed, "label"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "img"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "label"))


    print("Preprocessing training set...")
    pbar = tqdm(total = UNTIL_IDX - CONTINUE_AT_IDX)
    #for idx, (img, mask) in enumerate(train_loader.skip(CONTINUE_AT_IDX).take(UNTIL_IDX)):

    print(train_loader)
    for idx in range(CONTINUE_AT_IDX, UNTIL_IDX):
        args = train_loader[idx]
        img, mask = get_datapoint_train(args, config)
        pbar.update(1)
        save_datapoint(config.dataset_train_preprocessed, idx, img.numpy(), mask.numpy())
    pbar.close()

    if PREPROCESS_EVAL:        
        print("Preprocessing evaluation set...")
        idx = 0
        for seq_i in range(1, 9):
            test_loader =  Ego2HandsData(config, mode = "test_seg", seq_i = seq_i) # TODO expand to more sequences
            pbar = tqdm(total=test_loader.__len__())
            for args in test_loader:
                img, mask = get_datapoint_test(args, config)
                save_datapoint(config.dataset_eval_preprocessed, idx, img.numpy(), mask.numpy())
                pbar.update(1)
                idx += 1
            pbar.close()
    
    print("Done")




def save_datapoint(path, idx, img, mask):
    filename_img = os.path.join(path, "img", f"img_{idx:05}.png")
    filename_mask = os.path.join(path, "label", f"mask_{idx:05}.png")
    cv2.imwrite(filename_img, img)
    cv2.imwrite(filename_mask, mask)



def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def reformat_single(img, mask, shape):
    red, green, blue = tf.split(img, num_or_size_splits= 3, axis=2)
    img = tf.concat([blue, green, red], 2) # to BGR
    #img = tf.cast(img, tf.float32) / 255.0 # normalize
    img = tf.image.resize(img, shape[:2])
    mask = tf.image.resize(tf.expand_dims(mask, 2), shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return img, tf.squeeze(mask)

def get_datapoint_train(args, config):
    return reformat_single(args[1], args[3], config.input_shape)
    
def get_datapoint_test(args, config):
    return reformat_single(args[0], args[2], config.input_shape)


if __name__=="__main__":
    main()
