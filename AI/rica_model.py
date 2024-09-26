from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.losses import BinaryFocalCrossentropy

# Data generator class for loading images and masks
class BrainMRISequence(Sequence):
    def __init__(self, image_filenames, mask_filenames, image_dir, mask_dir, batch_size, image_size=(256, 256)):
        self.image_filenames = image_filenames
        self.mask_filenames = mask_filenames
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image_filenames = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_mask_filenames = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        images = np.array([img_to_array(load_img(os.path.join(self.image_dir, f), color_mode="grayscale")) for f in batch_image_filenames])
        masks = np.array([img_to_array(load_img(os.path.join(self.mask_dir, f), color_mode="grayscale")) for f in batch_mask_filenames])

        images = images / 255.0  # Normalize to [0,1]
        masks = masks / 255.0  # Normalize to [0,1]

        return images, masks

# Set up directories
image_dir = '/Users/edt/Desktop/p-brain/AI/output_pngs_rica'
mask_dir = '/Users/edt/Desktop/p-brain/AI/output_pngs_rica'
batch_size = 10

# Get list of image and mask filenames
image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') and 'slice' in f]
mask_filenames = [f.replace('slice', 'mask') for f in image_filenames]

# Split the data into training and validation sets
train_image_filenames, val_image_filenames, train_mask_filenames, val_mask_filenames = train_test_split(
    image_filenames, mask_filenames, test_size=0.2, random_state=42
)

# Create training and validation generators
train_sequence = BrainMRISequence(train_image_filenames, train_mask_filenames, image_dir, mask_dir, batch_size)
val_sequence = BrainMRISequence(val_image_filenames, val_mask_filenames, image_dir, mask_dir, batch_size)

# Define U-Net model with input shape (256, 256, 1)
def unet_2d(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Create and compile the model
input_shape = (256, 256, 1)
model = unet_2d(input_shape)
model.compile(optimizer='adam', loss=BinaryFocalCrossentropy(), metrics=['accuracy'])
model.summary()

# Set up callbacks
callbacks = [
    ModelCheckpoint('rica_roi_model.keras', save_best_only=True)
]

# Train the model
model.fit(train_sequence, validation_data=val_sequence, epochs=100, callbacks=callbacks)
