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

        if config.input_edge:
            filename_edge = os.path.join(config.dataset_train_preprocessed, "edge", f"edge_{idx:05}.png")
            edge = cv2.imread(filename_edge, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(filename_img, cv2.IMREAD_GRAYSCALE)
            img = np.stack([edge, img], axis=2)
        else:
            img = cv2.imread(filename_img)

        x_train.append(img)
        y_train.append(cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE))

    print("Loading evaluation images...")
    for idx in tqdm(range(2000)):
        filename_img = os.path.join(config.dataset_eval_preprocessed, "img", f"img_{idx:05}.png")
        filename_mask = os.path.join(config.dataset_eval_preprocessed, "label", f"mask_{idx:05}.png")
        
        if config.input_edge:
            filename_edge = os.path.join(config.dataset_eval_preprocessed, "edge", f"edge_{idx:05}.png")
            edge = cv2.imread(filename_edge, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(filename_img, cv2.IMREAD_GRAYSCALE)
            img = np.stack([edge, img], axis=2)
        else:
            img = cv2.imread(filename_img)

        x_eval.append(img)
        y_eval.append(cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE))
    
    return x_train, y_train, x_eval, y_eval


# ------------------------- DATASET AND CONSTANTS -------------------------------------

def main():
    config_path = os.path.join("configs", "config_tf.yml")
    config = Config(config_path)

    CONTINUE_AT_IDX = 0
    UNTIL_IDX = 2000000
    PREPROCESS_EVAL = True
    SAVE_TRAIN = True
    SAVE_EDGE = config.input_edge
    INPUT_SHAPE = config.input_shape

    train_loader = Ego2HandsData(config, mode = "train_seg")

    UNTIL_IDX = min(train_loader.__len__()-1, UNTIL_IDX) # limit 

    ensure_dir(os.path.join(config.dataset_train_preprocessed, "img"))
    ensure_dir(os.path.join(config.dataset_train_preprocessed, "label"))
    ensure_dir(os.path.join(config.dataset_train_preprocessed, "edge"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "img"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "label"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "edge"))



    print("Preprocessing training set...")
    pbar = tqdm(total = UNTIL_IDX - CONTINUE_AT_IDX)
    #for idx, (img, mask) in enumerate(train_loader.skip(CONTINUE_AT_IDX).take(UNTIL_IDX)):

    print(train_loader)
    img = None
    mask = None
    edge = None
    for idx in range(CONTINUE_AT_IDX, UNTIL_IDX):
        args = train_loader[idx]
        img_, mask_ = (x.numpy() for x in get_datapoint_train(args, config))
        if SAVE_EDGE:
            img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            edge = get_edge(img_.astype(np.uint8))
        if SAVE_TRAIN:
            img = img_
            mask = mask_
        pbar.update(1)
        save_datapoint(config.dataset_train_preprocessed, idx, img, mask, edge)
    pbar.close()

    if PREPROCESS_EVAL:        
        print("Preprocessing evaluation set...")
        idx = 0
        for seq_i in range(1, 9):
            test_loader =  Ego2HandsData(config, mode = "test_seg", seq_i = seq_i) # TODO expand to more sequences
            pbar = tqdm(total=test_loader.__len__())
            for args in test_loader:
                img, mask = (x.numpy() for x in get_datapoint_test(args, config))
                if SAVE_EDGE:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edge = get_edge(img.astype(np.uint8))
                save_datapoint(config.dataset_eval_preprocessed, idx, img, mask, edge)
                pbar.update(1)
                idx += 1
            pbar.close()
    print("Done")


def get_edge(img):
    edge = cv2.Canny(img, 50, 180)
    kernel = np.ones((3,3), np.uint8)
    edge = cv2.dilate(edge, kernel, iterations=1)
    edge = cv2.erode(edge, kernel, iterations=1)
    return edge



def save_datapoint(path, idx, img=None, mask=None, edge = None):
    if img is not None:
        filename_img = os.path.join(path, "img", f"img_{idx:05}.png")
        cv2.imwrite(filename_img, img)
    if mask is not None:
        filename_mask = os.path.join(path, "label", f"mask_{idx:05}.png")
        cv2.imwrite(filename_mask, mask)
    if edge is not None:
        filename_edge = os.path.join(path, "edge", f"edge_{idx:05}.png")
        cv2.imwrite(filename_edge, edge)




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
