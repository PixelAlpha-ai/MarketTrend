import numpy as np
import pandas as pd
from tensorflow.keras import metrics
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Conv1D, Flatten, Dense
from os import path

class InitialFeatureExtractor:
    def __init__(self, input_shape, num_initial_filters):
        self.model = Sequential([
            Conv2D(num_initial_filters, (4, 4), activation='relu', input_shape=input_shape, padding='valid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def get_feature_extractor(self):
        return Model(inputs=self.model.input, outputs=self.model.layers[-1].output)

def extract_intermediate_features(generator, feature_extractor):
    features = []
    labels = []
    for images, batch_labels in generator:
        batch_features = feature_extractor.predict(images)
        features.append(batch_features)
        labels.append(batch_labels)
    return np.vstack(features), np.hstack(labels)

class DataGenerator(Sequence):
    def __init__(self, csv_file, batch_size, folder_path, shuffle=True):
        self.df = pd.read_csv(csv_file)
        self.batch_size = batch_size
        self.folder_path = folder_path
        self.shuffle = shuffle
        self.image_shape = self._get_image_shape()
        self.on_epoch_end()

    def _get_image_shape(self):
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

class NewCNNModel:
    def __init__(self, input_shape, num_filters):
        self.model = Sequential([
            Conv1D(num_filters, 3, activation='relu', input_shape=input_shape),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(1, activation='sigmoid')  # Adjusted for binary classification
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

def create_and_train_new_model(train_features, train_labels, val_features, val_labels, input_shape, num_filters, class_weight, batch_size, path_data, model_name):
    new_cnn_model = NewCNNModel(input_shape=input_shape, num_filters=num_filters)
    new_cnn_model.model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    history = new_cnn_model.model.fit(
        train_features,
        train_labels,
        validation_data=(val_features, val_labels),
        epochs=50,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stopping]
    )

    path_trained_model = f"{path_data}\\model_{model_name}_filter_{num_filters}.h5"
    new_cnn_model.model.save(path_trained_model)
    return path_trained_model

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from objects_new import *

if __name__ == "__main__":
    path_data = 'C:\\data'
    model_name = '20240521_alternative_image'
    batch_size = 64
    num_initial_filters_list = [64]
    num_steps = 4

    master_train_folder = f'{path_data}\\data_training'
    master_validation_folder = f'{path_data}\\data_validation'
    master_testing_folder = f'{path_data}\\data_testing'
    if not os.path.exists(master_train_folder):
        os.mkdir(master_train_folder)
    if not os.path.exists(master_validation_folder):
        os.mkdir(master_validation_folder)
    if not os.path.exists(master_testing_folder):
        os.mkdir(master_testing_folder)

    symbols = ['BTCUSDT']
    timeframe_list = ['1h']

    master_train_log = pd.concat([pd.read_csv(f'{path_data}\\dataset_log_{symbol}_{timeframe}_training.csv') for symbol in symbols for timeframe in timeframe_list])
    master_validation_log = pd.concat([pd.read_csv(f'{path_data}\\dataset_log_{symbol}_{timeframe}_validation.csv') for symbol in symbols for timeframe in timeframe_list])
    master_testing_log = pd.concat([pd.read_csv(f'{path_data}\\dataset_log_{symbol}_{timeframe}_testing.csv') for symbol in symbols for timeframe in timeframe_list])

    path_master_train_log = f'{path_data}\\dataset_log_all_symbols_training.csv'
    master_train_log.to_csv(path_master_train_log, index=False)
    path_master_validation_log = f'{path_data}\\dataset_log_all_symbols_validation.csv'
    master_validation_log.to_csv(path_master_validation_log, index=False)
    path_master_testing_log = f'{path_data}\\dataset_log_all_symbols_testing.csv'
    master_testing_log.to_csv(path_master_testing_log, index=False)

    train_generator = DataGenerator(csv_file=path_master_train_log, batch_size=batch_size, folder_path=master_train_folder)
    val_generator = DataGenerator(csv_file=path_master_validation_log, batch_size=batch_size, folder_path=master_validation_folder)

    class_weight_dict = get_class_weight()

    images, _ = train_generator.__getitem__(0)
    input_shape = images.shape[1:]

    for num_initial_filters in num_initial_filters_list:
        feature_extractor = InitialFeatureExtractor(input_shape, num_initial_filters).get_feature_extractor()

        train_features, train_labels = extract_intermediate_features(train_generator, feature_extractor)
        val_features, val_labels = extract_intermediate_features(val_generator, feature_extractor)

        train_features = train_features.reshape((train_features.shape[0], train_features.shape[2], train_features.shape[3]))
        val_features = val_features.reshape((val_features.shape[0], val_features.shape[2], val_features.shape[3]))

        new_input_shape = (train_features.shape[1], train_features.shape[2])

        path_trained_model = create_and_train_new_model(train_features, train_labels, val_features, val_labels, new_input_shape, num_initial_filters, class_weight_dict, batch_size, path_data, model_name)

        cnn_model = tf.keras.models.load_model(path_trained_model)

        results = []
        for test_symbol in symbols:
            for test_timeframe in timeframe_list:
                test_folder = f'{path_data}\\data_testing_{test_symbol}_{test_timeframe}'
                test_log = f'{path_data}\\dataset_log_{test_symbol}_{test_timeframe}_testing.csv'
                test_generator = DataGenerator(csv_file=test_log, batch_size=batch_size, folder_path=test_folder)
                test_features, test_labels = extract_intermediate_features(test_generator, feature_extractor)

                test_features = test_features.reshape((test_features.shape[0], test_features.shape[2], test_features.shape[3]))

                score = cnn_model.evaluate(test_features, test_labels, verbose=0)
                predictions = cnn_model.predict(test_features, verbose=0)
                results.append([test_symbol, test_timeframe, round(score[1], 3)])
                print(f'Test accuracy for {test_symbol} {test_timeframe}:', round(score[1], 3))

        results_df = pd.DataFrame(results, columns=['Symbol', 'Timeframe', 'Accuracy'])
        results_df.to_csv(f'{path_data}\\results_{model_name}_{num_initial_filters}.csv', index=False)

        print(f'Results for {num_initial_filters} filters:')
        print(results_df)
