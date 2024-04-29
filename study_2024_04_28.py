""" found data leakge problem in previous code, needed to rerun studies to establish the best parameter"""
from objects import *

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf



# Main code - 2024-04-27 generate big model using all crypto study
if __name__ == "__main__":

    # Define paths to the data and log files
    path_data = 'C:\\data'

    # model parameters
    model_name = '20240428_sensitivity_analysis'
    batch_size = 64
    num_initial_filters_list = [16, 32, 64, 128]

    # Make three folders for training, validation, and testing
    master_train_folder = f'{path_data}\\data_training'
    master_validation_folder = f'{path_data}\\data_validation'
    master_testing_folder = f'{path_data}\\data_testing'
    if not os.path.exists(master_train_folder):
        os.mkdir(master_train_folder)
    if not os.path.exists(master_validation_folder):
        os.mkdir(master_validation_folder)
    if not os.path.exists(master_testing_folder):
        os.mkdir(master_testing_folder)

    # Parameters
    symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SOLUSDT', 'NEARUSDT', 'LINKUSDT']
    # symbols = ['BTCUSDT', 'SOLUSDT']
    # symbols = ['BTCUSDT']
    # timeframe_list = ['1h', '4h']
    timeframe_list = ['1h']

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
    for symbol in symbols:
        for timeframe in timeframe_list:
            train_folder = f'{path_data}\\data_training_{symbol}_{timeframe}'
            validation_folder = f'{path_data}\\data_validation_{symbol}_{timeframe}'
            testing_folder = f'{path_data}\\data_testing_{symbol}_{timeframe}'

            # Now copy the files from the subfolders to the master folder
            for file in os.listdir(train_folder):
                shutil.copy(f'{train_folder}\\{file}', master_train_folder)
            for file in os.listdir(validation_folder):
                shutil.copy(f'{validation_folder}\\{file}', master_validation_folder)
            for file in os.listdir(testing_folder):
                shutil.copy(f'{testing_folder}\\{file}', master_testing_folder)

    # Initialize the DataGenerators
    path_master_train_log = f'{path_data}\\dataset_log_all_symbols_training.csv'
    path_master_validation_log = f'{path_data}\\dataset_log_all_symbols_validation.csv'
    path_master_testing_log = f'{path_data}\\dataset_log_all_symbols_testing.csv'

    master_train_folder = f'{path_data}\\data_training'
    master_validation_folder = f'{path_data}\\data_validation'
    master_testing_folder = f'{path_data}\\data_testing'
    #

    ### If only use BTCUSDT
    # path_master_train_log = f'{path_data}\\dataset_log_BTCUSDT_1h_training.csv'
    # path_master_validation_log = f'{path_data}\\dataset_log_BTCUSDT_1h_validation.csv'
    # master_train_folder = f'{path_data}\\data_training_BTCUSDT_1h'
    # master_validation_folder = f'{path_data}\\data_validation_BTCUSDT_1h'

    train_generator = DataGenerator(csv_file=path_master_train_log, batch_size=batch_size, folder_path=master_train_folder)
    val_generator = DataGenerator(csv_file=path_master_validation_log, batch_size=batch_size, folder_path=master_validation_folder)

    # Extract the labels from train_generator
    # train_labels = np.array([row['label'] for _, row in train_generator.df.iterrows()])
    class_weight_dict = get_class_weight()

    # Load one batch of data to get the input shape
    images, _ = train_generator.__getitem__(0)
    input_shape = images.shape[1:]


    # Loop through the different number of initial filters
    for num_initial_filters in num_initial_filters_list:

        # Train the model
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

        # train the model
        history = cnn_model.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=50,
            use_multiprocessing=False,
            workers=1,
            callbacks=[early_stopping],
            class_weight=class_weight_dict,  # Use the manually assigned class weights here
        )

        # Save the model
        path_trained_model = f"{path_data}\\model_{model_name}_filter_{num_initial_filters}.h5"
        cnn_model.model.save(path_trained_model)



        ### Test the model
        # load the trained model h5
        cnn_model = tf.keras.models.load_model(path_trained_model)

        # Parameters
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SOLUSDT', 'NEARUSDT', 'LINKUSDT']
        symbols = ['BTCUSDT']
        timeframe_list = ['1h', '4h']
        timeframe_list = ['1h']

        # run individual tests
        results = []
        for test_symbol in symbols:
            for test_timeframe in timeframe_list:
                test_folder = f'{path_data}\\data_testing_{test_symbol}_{test_timeframe}'
                test_log = f'{path_data}\\dataset_log_{test_symbol}_{test_timeframe}_testing.csv'
                test_generator = DataGenerator(csv_file=test_log, batch_size=batch_size, folder_path=test_folder)
                score = cnn_model.evaluate(test_generator, verbose=0)
                predictions = cnn_model.predict(test_generator, verbose=0)
                results.append([test_symbol, test_timeframe, round(score[1], 3)])
                print(f'Test accuracy for {test_symbol} {test_timeframe}:', round(score[1], 3))

        # convert the results to a dataframe
        results_df = pd.DataFrame(results, columns=['Symbol', 'Timeframe', 'Accuracy'])
        results_df.to_csv(f'{path_data}\\results_{model_name}_{num_initial_filters}.csv', index=False)

        # print the results
        print(f'Results for {num_initial_filters} filters:')
        print(results_df)


