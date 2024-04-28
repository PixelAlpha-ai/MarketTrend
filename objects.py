import numpy as np
import pandas as pd
from tensorflow.keras import metrics
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from os import path


def get_class_weight():
    # Manual weights
    weight_for_0 = 1.0  # Weight for class 0
    weight_for_1 = 1.0  # Increase the weight for class 1 to emphasize it more during training

    return {0: weight_for_0, 1: weight_for_1}

class DataGenerator(Sequence):
    def __init__(self, csv_file, batch_size, folder_path, shuffle=True):
        self.df = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.shuffle = shuffle
        self.image_shape = self._get_image_shape()
        self.on_epoch_end()

    def _get_image_shape(self):
        # Load a single image to determine the shape
        sample_image_path = self.df.iloc[0]['filename'] + '.npy'
        sample_image = np.load(path.join(self.folder_path, sample_image_path))
        return sample_image.shape + (1,)  # Adding the channel dimension

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch = [self.df.iloc[k] for k in indices]

        images = np.array(
            [np.load(path.join(self.folder_path, row['filename'] + '.npy')).reshape(self.image_shape) for row in batch])
        labels = np.array([row['label'] for row in batch])

        return images, labels


class CNNModel:
    def __init__(self, input_shape, num_initial_filters):
        self.model = Sequential([
            Conv2D(num_initial_filters, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 1)),
            Conv2D(num_initial_filters*2, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 1)),
            Conv2D(num_initial_filters*4, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 1)),
            Flatten(),
            Dense(1, activation='sigmoid')  # Adjusted for binary classification
        ])
        self.model.compile(optimizer='adam',
                           loss='binary_crossentropy',  # Adjusted for binary classification
                           metrics=['accuracy',
                                    metrics.Precision(name='precision'),
                                    metrics.Recall(name='recall'),
                                    metrics.AUC(name='auc')])




