import cv2
import tensorflow as tf
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import Config
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from preprocess_dataset import get_preprocessed_datasets


# ------------------------- DATASET AND CONSTANTS -------------------------------------

config_path = os.path.join("configs", "config_tf.yml")
config = Config(config_path)


#elf.img_h, self.img_w = 

PRED_BATCH_INTERVAL = 10
TRAIN_LENGTH = 1000
BATCH_SIZE = config.batch_size
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CLASSES = 1

INPUT_SHAPE = [128, 128, 3]

EPOCHS = config.max_iter_seg
VAL_SUBSPLITS = 5
VALIDATION_STEPS =TRAIN_LENGTH//BATCH_SIZE//VAL_SUBSPLITS

x_train, y_train, x_eval, y_eval = get_preprocessed_datasets(config, TRAIN_LENGTH)

normalize = lambda x: tf.cast(x, tf.float32)/255.0
to_tensor = lambda x: tf.constant(x)
x_train = [normalize(x) for x in x_train]
x_eval = [normalize(x) for x in x_eval]
y_train = [tf.convert_to_tensor(x) for x in y_train]
y_eval = [tf.convert_to_tensor(x) for x in y_eval]

# convert list of tensors to [n, *[0].shape] tensor
x_train = tf.stack(x_train)
y_train = tf.stack(y_train)
x_eval = tf.stack(x_eval)
y_eval = tf.stack(y_eval)



print("Loaded training set: ", len(x_train))


def display(display_list, block = False):
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        #cv2.imshow(title[i], display_list[i].numpy())
        cv2.imwrite(f"output_{i}.png", display_list[i].numpy())
    #if block:
    #    cv2.waitKey(0)
    #else:
    #    cv2.waitKey(1)


# for sample_image_batch, sample_mask_batch in train_loader.take(2):
    # sample_image, sample_mask = sample_image_batch[0], sample_mask_batch[0]
    # display([tf.cast(sample_image*255, tf.uint8), sample_mask*50], block=False)
# 
# 
# print("Press space to continue...")
# for sample_image, sample_mask in zip(x_train[:2], y_train[:2]):
    # sample_image = tf.cast(sample_image*255, tf.uint8)
    # sample_mask *= 50
    # display([sample_image, sample_mask], block=True)

# ---------------------------------------- MODEL DEFINITION -------------------------------------------------
def upsample(filters, size, apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        result.add(tf.keras.layers.ReLU())
        return result

def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE)

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
down_stack.trainable = False

up_stack = [
    upsample(512, 3),  # 4x4 -> 8x8
    upsample(256, 3),  # 8x8 -> 16x16
    upsample(128, 3),  # 16x16 -> 32x32
    upsample(64, 3),   # 32x32 -> 64x64
]

model = unet_model(output_channels=OUTPUT_CLASSES)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])


def create_mask_indices(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def create_mask(pred_mask):
    pred_mask[pred_mask>=0.5] = 1
    pred_mask[pred_mask<0.5] = 0
    return tf.squeeze(pred_mask)

def show_predictions(dataset=None, num=1, block=False):
    if dataset:
        for image, mask in dataset:
            pred_mask = model.predict(image)
            pred_mask = tf.cast(create_mask(pred_mask), tf.uint8) * 50
            display([tf.cast(image[0]*255, tf.uint8), mask[0]*50, pred_mask], block=block)
    else:
        prediction = model.predict(sample_image[tf.newaxis, ...])
        predicted_mask = tf.cast(create_mask(prediction), tf.uint8) * 50
        display([sample_image, sample_mask, predicted_mask], block=block)
                    
#show_predictions(block=True)

logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions(num = 3)

class TensorBoardImageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        imgs = []
        gts = []
        preds = []

        num = 3

        imgs = x_eval[:num]
        gts = y_eval[:num]
        gts = tf.expand_dims(gts, -1)
        preds = model.predict(imgs)
        preds = tf.cast(create_mask(preds), tf.uint8) * 50
        preds = tf.expand_dims(preds, -1)
        imgs = tf.cast(imgs*255.0, tf.uint8)
        gts = tf.cast(gts*50, tf.uint8)

        file_writer = tf.summary.create_file_writer(logdir)
        with file_writer.as_default():
            tf.summary.image("imgs", imgs, max_outputs = num, step=epoch)
            tf.summary.image("gts", gts, max_outputs = num, step=epoch)
            tf.summary.image("preds", preds,max_outputs = num, step=epoch)




model_history = model.fit(x_train, y_train,
                          epochs=EPOCHS,
                          batch_size = config.batch_size,
                          validation_data=(x_eval, y_eval),
                          callbacks=[tf.keras.callbacks.ModelCheckpoint(
                                                'simple_tf/model_checkpoint/', monitor='val_loss', verbose=0, 
                                                save_best_only=True,save_weights_only=True, mode='min'), 
                                    tf.keras.callbacks.TensorBoard(log_dir=logdir),
                                    #DisplayCallback(),
                                    TensorBoardImageCallback()
                          ])

model.save('simple_tf/last/')
