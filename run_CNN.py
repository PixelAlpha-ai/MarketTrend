import numpy as np
import pandas as pd
import tensorflow as tf
import os
import shutil
from tensorflow.keras import metrics
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from os import path


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

# def get_class_weight(y_train):
#     # Automatic weights
#     # Assuming y_train is your labels array for training data
#     # and it's a numpy array. Adjust the labels as necessary.
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     return {i : class_weights[i] for i in range(len(class_weights))}

def get_class_weight():
    # Manual weights
    # You can adjust these weights based on your specific needs
    # For example, increasing the weight of the positive class to make it more significant during training
    weight_for_0 = 1.0  # Weight for class 0
    weight_for_1 = 1.0  # Increase the weight for class 1 to emphasize it more during training

    return {0: weight_for_0, 1: weight_for_1}




# Main code - 2024-04-27 generate big model using all crypto study
if __name__ == "__main__":

    # Define paths to the data and log files
    path_data = 'C:\\data'

    # Make three folders for training, validation, and testing
    train_folder = f'{path_data}\\data_training'
    validation_folder = f'{path_data}\\data_validation'
    testing_folder = f'{path_data}\\data_testing'

    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(validation_folder):
        os.mkdir(validation_folder)
    if not os.path.exists(testing_folder):
        os.mkdir(testing_folder)

    # Parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SOLUSDT', 'NEARUSDT', 'LINKUSDT']
    symbols = ['BTCUSDT', 'SOLUSDT']
    timeframe_list = ['1h', '4h']

    # Combine all training logs and validation logs into one master log (one for training and one for validation)
    master_train_log = pd.concat([pd.read_csv(f'{path_data}\\dataset_log_{symbol}_{timeframe}_training.csv') for symbol in symbols for timeframe in timeframe_list])
    master_validation_log = pd.concat([pd.read_csv(f'{path_data}\\dataset_log_{symbol}_{timeframe}_validation.csv') for symbol in symbols for timeframe in timeframe_list])
    master_testing_log = pd.concat([pd.read_csv(f'{path_data}\\dataset_log_{symbol}_{timeframe}_testing.csv') for symbol in symbols for timeframe in timeframe_list])

    # Save the master logs
    path_master_train_log = f'{path_data}\\dataset_log_all_symbols_training.csv'
    master_train_log.to_csv(path_master_train_log, index=False)
    path_master_validation_log = f'{path_data}\\dataset_log_all_symbols_validation.csv'
    master_validation_log.to_csv(path_master_validation_log, index=False)
    path_master_testing_log = f'{path_data}\\dataset_log_all_symbols_testing.csv'
    master_testing_log.to_csv(path_master_testing_log, index=False)


    # Now combine the files into one folder, do this by looping through all the subfolders
    # for symbol in symbols:
    #     for timeframe in timeframe_list:
    #         train_folder = f'{path_data}\\data_training_{symbol}_{timeframe}'
    #         validation_folder = f'{path_data}\\data_validation_{symbol}_{timeframe}'
    #         testing_folder = f'{path_data}\\data_testing_{symbol}_{timeframe}'
    #
    #         # Now copy the files from the subfolders to the master folder
    #         for file in os.listdir(train_folder):
    #             shutil.copy(f'{train_folder}\\{file}', f'{path_data}\\data_training')
    #         for file in os.listdir(validation_folder):
    #             shutil.copy(f'{validation_folder}\\{file}', f'{path_data}\\data_validation')
    #         for file in os.listdir(testing_folder):
    #             shutil.copy(f'{testing_folder}\\{file}', f'{path_data}\\data_testing')

    # Initialize the DataGenerators
    batch_size = 64
    train_generator = DataGenerator(csv_file=path_master_train_log, batch_size=batch_size, folder_path=train_folder)
    val_generator = DataGenerator(csv_file=path_master_validation_log, batch_size=batch_size, folder_path=validation_folder)

    # Extract the labels from train_generator
    train_labels = np.array([row['label'] for _, row in train_generator.df.iterrows()])
    class_weight_dict = get_class_weight()

    # Load one batch of data to get the input shape
    images, _ = train_generator.__getitem__(0)
    input_shape = images.shape[1:]

    # Train the model
    num_initial_filters = 64
    cnn_model = CNNModel(input_shape=input_shape, num_initial_filters=num_initial_filters)
    cnn_model.model.summary()

    # Set the early stop
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        min_delta=0.001,  # Minimum change to qualify as an improvement
        patience=3,  # Number of epochs with no improvement after which training will be stopped
        verbose=1,  # Logging level
        mode='min',  # The direction is automatically inferred if not set
        restore_best_weights=True
        # Whether to restore model weights from the epoch with the best value of the monitored quantity
    )

    # Save the model
    name_trained_model = f"{path_data}\\model_20240427_all_symbols.h5"
    cnn_model.model.save(name_trained_model)

    try:
        history = cnn_model.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            use_multiprocessing=False,
            workers=1,
            callbacks=[early_stopping],
            class_weight=class_weight_dict,  # Use the manually assigned class weights here
        )

        # loop through a range of symbols and timeframes to test, organize the results in a table for easy copy-paste

        # generate a table of results
        results = []

        for test_symbol in symbols:
            for test_timeframe in timeframe_list:
                test_folder = f'{path_data}\\data_testing_{test_symbol}_{test_timeframe}'
                test_log = f'dataset_log_{test_symbol}_{test_timeframe}_testing.csv'
                test_generator = DataGenerator(csv_file=test_log, batch_size=batch_size, folder_path=test_folder)
                score = cnn_model.model.evaluate(test_generator, verbose=0)
                predictions = cnn_model.model.predict(test_generator, verbose=0)
                results.append([test_symbol, test_timeframe, round(score[1], 3)])
                print(f'Test accuracy for {test_symbol} {test_timeframe}:', round(score[1], 3))

        # convert the results to a dataframe
        results_df = pd.DataFrame(results, columns=['Symbol', 'Timeframe', 'Accuracy'])
        results_df.to_csv(f'{path_data}\\results_20240427_all_symbols.csv', index=False)
        print(results_df)


    except Exception as e:
        print(f"An exception occurred: {e}")
    finally:
        tf.keras.backend.clear_session()



# # Main code 2024-04-25 study
# if __name__ == "__main__":
#
#     # Define paths to the data and log files
#     path_data = 'C:\\data'
#
#     # Training set
#     symbol_training = 'BTCUSDT'
#     symbol_training = 'ETHUSDT'
#     symbol_training = 'BNBUSDT'
#     symbol_training = 'DOGEUSDT'
#     symbol_training = 'SOLUSDT'
#     symbol_training = 'NEARUSDT'
#     symbol_training = 'LINKUSDT'
#     time_frame_training = '4h'
#     train_log = f'dataset_log_{symbol_training}_{time_frame_training}_training.csv'
#     train_folder = f'{path_data}\\data_training_{symbol_training}_{time_frame_training}'
#
#     # Validation set
#     validation_folder = f'{path_data}\\data_validation_{symbol_training}_{time_frame_training}'
#     validation_log = f'dataset_log_{symbol_training}_{time_frame_training}_validation.csv'
#
#
#     # Initialize the DataGenerators
#     batch_size = 64
#     train_generator = DataGenerator(csv_file=train_log, batch_size=batch_size, folder_path=train_folder)
#     val_generator = DataGenerator(csv_file=validation_log, batch_size=batch_size, folder_path=validation_folder)
#
#     # Extract the labels from train_generator
#     train_labels = np.array([row['label'] for _, row in train_generator.df.iterrows()])
#     class_weight_dict = get_class_weight()
#
#     # Load one batch of data to get the input shape
#     images, _ = train_generator.__getitem__(0)
#     input_shape = images.shape[1:]
#
#     # Train the model
#     num_initial_filters = 64
#     cnn_model = CNNModel(input_shape=input_shape, num_initial_filters=num_initial_filters)
#     cnn_model.model.summary()
#
#     # Set the early stop
#     early_stopping = EarlyStopping(
#         monitor='val_loss',  # Metric to monitor
#         min_delta=0.001,  # Minimum change to qualify as an improvement
#         patience=3,  # Number of epochs with no improvement after which training will be stopped
#         verbose=1,  # Logging level
#         mode='min',  # The direction is automatically inferred if not set
#         restore_best_weights=True
#         # Whether to restore model weights from the epoch with the best value of the monitored quantity
#     )
#
#     # Save the model
#     name_trained_model = f"{path_data}\\model_{symbol_training}_{time_frame_training}.h5"
#     cnn_model.model.save(name_trained_model)
#
#     try:
#         history = cnn_model.model.fit(
#             train_generator,
#             validation_data=val_generator,
#             epochs=50,
#             use_multiprocessing=False,
#             workers=1,
#             callbacks=[early_stopping],
#             class_weight=class_weight_dict,  # Use the manually assigned class weights here
#         )
#
#         # loop through a range of symbols and timeframes to test
#         for test_symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SOLUSDT', 'NEARUSDT', 'LINKUSDT']:
#             for test_timeframe in ['1h', '4h']:
#                 test_folder = f'{path_data}\\data_testing_{test_symbol}_{test_timeframe}'
#                 test_log = f'dataset_log_{test_symbol}_{test_timeframe}_testing.csv'
#                 test_generator = DataGenerator(csv_file=test_log, batch_size=batch_size, folder_path=test_folder)
#                 score = cnn_model.model.evaluate(test_generator, verbose=0)
#                 predictions = cnn_model.model.predict(test_generator, verbose=0)
#                 # print(f'Test loss for {test_symbol} {test_timeframe}:', score[0])
#                 print(f'Test accuracy for {test_symbol} {test_timeframe}:', round(score[1], 3))
#
#     except Exception as e:
#         print(f"An exception occurred: {e}")
#     finally:
#         tf.keras.backend.clear_session()
#
#     ### Post-processing to show resuslts
#     # Initialize lists to hold actual and predicted labels
#     # actual_labels = []
#     # predicted_labels = []
#     #
#     # # Iterate through the test set
#     # for i in range(len(test_generator)):
#     #     x_test, y_test = test_generator[i]
#     #     predictions = cnn_model.model.predict(x_test)
#     #     predicted_classes = np.round(predictions).astype('int').flatten()
#     #     actual_labels.extend(y_test)
#     #     predicted_labels.extend(predicted_classes)
#     #
#     # # Convert lists to numpy arrays for consistency
#     # actual_labels = np.array(actual_labels)
#     # predicted_labels = np.array(predicted_labels)
#     #
#     # # Compute the confusion matrix
#     # cm = confusion_matrix(actual_labels, predicted_labels)
#     #
#     # # Plot confusion matrix
#     # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#     # plt.xlabel('Predicted Label')
#     # plt.ylabel('True Label')
#     # plt.title('Confusion Matrix')
#     # plt.show()
#     #
#     # # Optionally, print a classification report for more metrics (precision, recall, F1-score)
#     # print(classification_report(actual_labels, predicted_labels))