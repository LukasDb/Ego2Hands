import multiprocessing as mp
from tqdm import tqdm
import tensorflow as tf
import cv2
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import Config
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

INPUT_SHAPE = [128, 128, 3]
EPOCHS = 20
VAL_SUBSPLITS = 5
OUTPUT_CLASSES = 1
BUFFER_SIZE = 1000
PRED_BATCH_INTERVAL = 10

CONTINUE_AT_IDX = 3000
UNTIL_IDX = 2000 # actually how many! NOT until


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
    for idx in tqdm(range(250)):
        filename_img = os.path.join(config.dataset_eval_preprocessed, "img", f"img_{idx:05}.png")
        filename_mask = os.path.join(config.dataset_eval_preprocessed, "label", f"mask_{idx:05}.png")
        x_eval.append(cv2.imread(filename_img))
        y_eval.append(cv2.imread(filename_mask, cv2.IMREAD_GRAYSCALE))
    
    return x_train, y_train, x_eval, y_eval


# ------------------------- DATASET AND CONSTANTS -------------------------------------

def main():
    config_path = os.path.join("configs", "config_tf.yml")
    config = Config(config_path)

    n_imgs, train_loader, test_loader = get_tf_datasets(config)

    TRAIN_LENGTH = n_imgs
    BATCH_SIZE = config.batch_size
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    VALIDATION_STEPS = n_imgs//BATCH_SIZE//VAL_SUBSPLITS


    ensure_dir(os.path.join(config.dataset_train_preprocessed, "img"))
    ensure_dir(os.path.join(config.dataset_train_preprocessed, "label"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "img"))
    ensure_dir(os.path.join(config.dataset_eval_preprocessed, "label"))

    print("Preprocessing training set...")
    pbar = tqdm(total = TRAIN_LENGTH)
    for idx, (img, mask) in enumerate(train_loader.skip(CONTINUE_AT_IDX).take(UNTIL_IDX)):
        pbar.update(1)
        save_datapoint(config.dataset_train_preprocessed, idx+CONTINUE_AT_IDX, img.numpy(), mask.numpy())
    pbar.close()

        
    print("Preprocessing evaluation set...")
    pbar = tqdm(total=250)
    for idx, (img, mask) in enumerate(test_loader):
        save_datapoint(config.dataset_eval_preprocessed, idx, img.numpy(), mask.numpy())
        pbar.update(1)
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

def reformat_single(img, mask):
    red, green, blue = tf.split(img, num_or_size_splits= 3, axis=2)
    img = tf.concat([blue, green, red], 2) # to BGR
    #img = tf.cast(img, tf.float32) / 255.0 # normalize
    img = tf.image.resize(img, INPUT_SHAPE[:2])
    mask = tf.image.resize(tf.expand_dims(mask, 2), INPUT_SHAPE[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return (img, tf.squeeze(mask))

def get_datapoint(*args):
    return reformat_single(args[1], args[3])
    
def get_datapoint_test(*args):
    return reformat_single(args[0], args[2])

def get_tf_datasets(config):
    hand_dataset_train = lambda : Ego2HandsData(config, mode = "train_seg")
    hand_dataset_test = lambda : Ego2HandsData(config, mode = "test_seg", seq_i = 1) # TODO expand to more sequences

    temp = hand_dataset_train()
    temp_test = hand_dataset_test()
    h, w = (temp.img_h, temp.img_w) # 288, 512
    output_signature = (
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w, 2 if config.energy else 1), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w), dtype=tf.uint8),
        tf.TensorSpec(shape=(h//2, w//2), dtype=tf.uint8),
        tf.TensorSpec(shape=(h//4, w//4), dtype=tf.uint8),
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h//2, w//2, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h//4, w//4, 3), dtype=tf.float32),
    )
    output_signature_test = (
        tf.TensorSpec(shape=(h, w, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w, 2 if config.energy else 1), dtype=tf.float32),
        tf.TensorSpec(shape=(h, w), dtype=tf.uint8),
    )

    train_loader = tf.data.Dataset.from_generator(hand_dataset_train, output_signature=output_signature) \
        .map(get_datapoint, num_parallel_calls=tf.data.AUTOTUNE).cache()

    test_loader = tf.data.Dataset.from_generator(hand_dataset_test, output_signature=output_signature_test) \
        .map(get_datapoint_test, num_parallel_calls=tf.data.AUTOTUNE).cache()

    return temp.__len__(), train_loader, test_loader

if __name__=="__main__":
    main()
