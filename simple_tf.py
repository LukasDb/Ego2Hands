import cv2
import tensorflow as tf
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import Config
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from preprocess_dataset import get_preprocessed_datasets
import tensorflow_addons as tfa

# ------------------------- DATASET AND CONSTANTS -------------------------------------

config_path = os.path.join("configs", "config_tf.yml")
config = Config(config_path)

logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")

#elf.img_h, self.img_w = 

PRED_BATCH_INTERVAL = 10
TRAIN_LENGTH = 78253
VAL_LENGTH = 2000
BATCH_SIZE = config.batch_size
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
VALIDATION_STEPS =VAL_LENGTH//BATCH_SIZE
OUTPUT_CLASSES = 1

INPUT_SHAPE = config.input_shape

EPOCHS = config.max_iter_seg

x_train, y_train, x_eval, y_eval = get_preprocessed_datasets(config, TRAIN_LENGTH)
x_train = np.array(x_train)
n_batches = STEPS_PER_EPOCH

x_train = np.reshape(x_train[:n_batches * BATCH_SIZE], (n_batches, BATCH_SIZE, *x_train[0].shape)) # reshape to a list of batches
y_train = np.reshape(y_train[:n_batches * BATCH_SIZE], (n_batches, BATCH_SIZE, *y_train[0].shape)) # reshape to a list of batches

x_eval = np.reshape(x_eval[:VALIDATION_STEPS * BATCH_SIZE], (VALIDATION_STEPS, BATCH_SIZE, *(x_eval[0].shape))) # reshape to a list of batches
y_eval = np.reshape(y_eval[:VALIDATION_STEPS * BATCH_SIZE], (VALIDATION_STEPS, BATCH_SIZE, *y_eval[0].shape)) # reshape to a list of batches

def train_generator(num = n_batches):
    ind = 0
    while (ind < num):
        yield (tf.convert_to_tensor(x_train[ind], dtype = tf.float16)/255.0, 
               tf.convert_to_tensor(y_train[ind], dtype=tf.float16))
        ind += 1
        if ind == n_batches:
            ind = 0


def test_generator(num = None, random = False):
    ind = 0 if not random else np.random.randint(0, VALIDATION_STEPS)
    output = 0
    while (output < num) if num is not None else True:
        yield (tf.convert_to_tensor(x_eval[ind], dtype=tf.float16)/255.0, 
               tf.convert_to_tensor(y_eval[ind], dtype=tf.float16))
        ind += 1
        output += 1
        if ind == VALIDATION_STEPS:
            ind = 0

print("Loaded training set: ", len(x_train)*config.batch_size)


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
def upsample(filters, size, apply_dropout=False, activation='relu'):
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

        if activation == 'relu':
            result.add(tf.keras.layers.ReLU())
        elif activation == 'sigmoid':
            result.add(tf.keras.layers.Activation(activation='sigmoid'))
        return result

def downsample(filters, size, apply_dropout=False, activation='relu', ):
        initializer = tf.random_normal_initializer(0., 0.02)

        result = tf.keras.Sequential()
        result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

        result.add(tf.keras.layers.BatchNormalization())

        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))

        if activation == 'relu':
            result.add(tf.keras.layers.ReLU())
        elif activation == 'sigmoid':
            result.add(tf.keras.layers.Activation(activation='sigmoid'))
        return result

class BaseModel(tf.keras.Model):
    def __init__(self, filter_size = 3):
        super().__init__()
        self.dslayers = [
            downsample(32, filter_size, apply_dropout = True),
            downsample(64, filter_size, apply_dropout = True),
            downsample(128, filter_size, apply_dropout = True),
            downsample(256, filter_size, apply_dropout = True),
            downsample(512, filter_size, apply_dropout = True),
        ]
    
    def call(self, x, train=False):
        outputs = []
        for layer in self.dslayers:
            x = layer(x)
            outputs.append(x)
        return outputs



def unet_model(output_channels:int):
    inputs = tf.keras.layers.Input(shape=(*INPUT_SHAPE[:2], 2)) # h, w, 2 stacked gray + edge

    # if not config.input_edge
    # base_model = tf.keras.applications.MobileNetV2(input_shape=INPUT_SHAPE, include_top=False)
    # # Use the activations of these layers
    # layer_names = [
        # 'block_1_expand_relu',   # 64x64
        # 'block_3_expand_relu',   # 32x32
        # 'block_6_expand_relu',   # 16x16
        # 'block_13_expand_relu',  # 8x8
        # 'block_16_project',      # 4x4
    # ]
    # base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    # # Create the feature extraction model
    # down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    # down_stack.summary()
    # down_stack.trainable = False
    filter_size = 3
    down_stack = BaseModel(filter_size)

    up_stack = [
        upsample(512, filter_size, apply_dropout = True),
        upsample(256, filter_size, apply_dropout = True),
        upsample(128, filter_size, apply_dropout = True),
        upsample(64, filter_size, apply_dropout = True),
    ]

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
        filters=32, kernel_size=filter_size, strides=2,
        padding='same', activation = 'sigmoid')  #64x64 -> 128x128

    x = last(x)

    x = tf.keras.layers.Conv2D(16, 5, padding='same')(x)
    x = tf.keras.layers.Conv2D(output_channels, 5, padding='same', activation='sigmoid')(x)

    return tf.keras.Model(inputs=inputs, outputs=x)



model = unet_model(output_channels=OUTPUT_CLASSES)
# loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
model.compile(optimizer='adam',
            loss=loss,
            metrics=['accuracy'])

model.summary()


def create_mask_indices(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def create_mask(pred_mask):
    pred_mask[pred_mask>=0.5] = 1
    pred_mask[pred_mask<0.5] = 0
    return tf.squeeze(pred_mask)

def show_predictions():
    imgs = []
    gts = []
    preds = []

    num = 3
    data = [(x,y) for x,y in test_generator(1, random = True)][0]
    imgs = data[0] # pick x
    gts = data[1] # pick y
    
    gts = tf.expand_dims(gts, -1) # add gchannel to grayscale
    preds = model.predict(imgs)

    preds = tf.cast(create_mask(preds), tf.uint8)
    #preds = [postprocess(pred) for pred in preds]
    preds = preds * 50
    preds = tf.expand_dims(preds, -1) # add channel to grayscale

    imgs = tf.cast(imgs[..., 1]*255.0, tf.uint8) # pick grayscale not edges
    imgs = tf.expand_dims(imgs, -1) # add channel to grayscale 
    gts = tf.cast(gts*100, tf.uint8)
    
    display([imgs[0], gts[0], preds[0]])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()

def visualize(img, mask):
    mask = tf.cast(mask*255, tf.float32)
    mask = tfa.image.median_filter2d(mask, filter_shape = (3,3))
    mask = tfa.image.gaussian_filter2d(mask, filter_shape = (3,3), sigma = 1.0)
    mask = tf.where(mask>1, 255, 0) # thresholding

    zero_mask = tf.zeros_like(mask)
    rgb_mask = tf.concat([mask, zero_mask, zero_mask], -1)
    rgb_mask = tf.cast(rgb_mask, tf.float32)

    rgb_img = tf.image.grayscale_to_rgb(img)
    #overlay = tfa.image.blend(rgb_img, rgb_mask, 0) # convert to rgb!
    overlay = tf.cast(rgb_img, tf.float32)/2. + rgb_mask/2.
    
    return mask, tf.cast(overlay, tf.uint8)


class TensorBoardImageCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        imgs = []
        gts = []
        preds = []

        num = 3
        data = [(x,y) for x,y in test_generator(1, random = True)][0]
        imgs = data[0]

        preds = model.predict(imgs)
        preds = tf.cast(create_mask(preds), tf.uint8)

        imgs = tf.cast(imgs[..., 1]*255.0, tf.uint8) # only pick grayscale

        imgs = tf.expand_dims(imgs, -1) # add channel to grayscale
        preds = tf.expand_dims(preds, -1)

        _, overlays = visualize(imgs[:num], preds[:num])

        preds = preds * 100
        
        gts = data[1]
        gts = tf.expand_dims(gts, -1)
        gts = tf.cast(gts*100, tf.uint8)

        file_writer = tf.summary.create_file_writer(logdir)
        with file_writer.as_default():
            tf.summary.image("imgs", imgs, max_outputs = num, step=epoch)
            tf.summary.image("gts", gts, max_outputs = num, step=epoch) # they never change
            tf.summary.image("preds", preds, max_outputs = num, step=epoch)
            tf.summary.image("Visualization", overlays, max_outputs = num, step=epoch)



model_history = model.fit(train_generator(),
                          epochs=EPOCHS,
                          batch_size = config.batch_size,
                          validation_data=test_generator(),
                          validation_batch_size = config.batch_size,
                          validation_steps = VALIDATION_STEPS,
                          steps_per_epoch = STEPS_PER_EPOCH,
                          callbacks=[tf.keras.callbacks.ModelCheckpoint(
                                                'simple_tf/model_checkpoint/', monitor='val_loss', verbose=0, 
                                                save_best_only=True,save_weights_only=True, mode='min'), 
                                    tf.keras.callbacks.TensorBoard(log_dir=logdir),
                                    #DisplayCallback(),
                                    TensorBoardImageCallback()
                          ])

model.save('simple_tf/last/')
