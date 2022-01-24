import cv2
import tensorflow as tf
from data_loaders.Ego2Hands_tf import Ego2HandsData
import os
from utils import Config
import matplotlib.pyplot as plt


# ------------------------- DATASET AND CONSTANTS -------------------------------------

config_path = os.path.join("configs", "config_tf.yml")
config = Config(config_path)
hand_dataset_train = lambda : Ego2HandsData(config, mode = "train_seg")
hand_dataset_test = lambda : Ego2HandsData(config, mode = "test_seg", seq_i = 1)

#elf.img_h, self.img_w = 
temp = hand_dataset_train()
temp_test = hand_dataset_test()
h, w = (temp.img_h, temp.img_w) # 288, 512


TRAIN_LENGTH = temp.__len__()
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
OUTPUT_CLASSES = 3

INPUT_SHAPE = [128, 128, 3]

EPOCHS = 20
VAL_SUBSPLITS = 5
VALIDATION_STEPS = temp_test.__len__()//BATCH_SIZE//VAL_SUBSPLITS



def get_datapoint(*args):
    img = tf.cast(args[1], tf.float32) / 255.0 # normalize
    img = tf.image.resize(img, INPUT_SHAPE[:2])
    mask = args[3]
    mask = tf.image.resize(tf.expand_dims(mask, 2), INPUT_SHAPE[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return (img, tf.squeeze(mask))

def get_datapoint_test(*args):
    img = tf.cast(args[0], tf.float32) / 255.0 # normalize
    img = tf.image.resize(img, INPUT_SHAPE[:2])
    mask = args[2]
    mask = tf.image.resize(tf.expand_dims(mask, 2), INPUT_SHAPE[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return (img, tf.squeeze(mask))

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
    .map(get_datapoint, num_parallel_calls=tf.data.AUTOTUNE) \
    .cache().batch(config.batch_size).shuffle(20).prefetch(tf.data.AUTOTUNE)

test_loader = tf.data.Dataset.from_generator(hand_dataset_test, output_signature=output_signature_test) \
     .map(get_datapoint_test, num_parallel_calls=tf.data.AUTOTUNE) \
    .cache().batch(1).prefetch(tf.data.AUTOTUNE)



def display(display_list, block = False):
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    for i in range(len(display_list)):
        cv2.imshow(title[i], display_list[i].numpy())
    cv2.waitKey(0 if block else 1)

for sample_image_batch, sample_mask_batch in train_loader.take(2):
    sample_image, sample_mask = sample_image_batch[0], sample_mask_batch[0]
    display([tf.cast(sample_image*255, tf.uint8), sample_mask*50], block=False)


print("Press space to continue...")
for sample_image_batch, sample_mask_batch in test_loader.take(2):
    sample_image, sample_mask = tf.cast(sample_image_batch[0]*255, tf.uint8), sample_mask_batch[0]*50
    display([sample_image, sample_mask], block=True)

print("Fitting....")


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

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])



def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            pred_mask = tf.cast(create_mask(pred_mask), tf.uint8) * 50
            display([tf.cast(image[0]*255, tf.uint8), mask[0]*50, pred_mask])
    else:
        prediction = model.predict(sample_image[tf.newaxis, ...])
        predicted_mask = tf.cast(create_mask(prediction), tf.uint8) * 50
        display([sample_image, sample_mask, predicted_mask])
                    

show_predictions()


class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))


model_history = model.fit(train_loader, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_loader,
                          callbacks=[DisplayCallback()])

loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

plt.figure()
plt.plot(model_history.epoch, loss, 'r', label='Training loss')
plt.plot(model_history.epoch, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()


show_predictions(test_loader, 10)