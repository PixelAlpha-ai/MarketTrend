""" Define a function to read a pandas dataframe of OHLCV data and to convert into 2D binary image"""
""" New style of 2D image - 4xN matrix, where rows represent the open, high, low, and close, and columns represent 
the candles, OHLC values are normalized by running ATH values"""


import pandas as pd
import datetime
import numpy as np
from PIL import Image
import talib
import os
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


def generate_2d_image(df_OHLC_session):
    """
    :param df_OHLC_session: the session data that contains OHLCV, and the running min/max for the plot range
    :return: the 2D binary image of the OHLC data
    """
    num_rows = 4
    num_cols = df_OHLC_session.shape[0]
    image_matrix = np.zeros((num_rows, num_cols))

    # Fill in the image with the normalized OHLC data
    for i, (idx, row) in enumerate(df_OHLC_session.iterrows()):
        image_matrix[0, i] = row['Open']
        image_matrix[1, i] = row['High']
        image_matrix[2, i] = row['Low']
        image_matrix[3, i] = row['Close']

    return image_matrix

def save_image(image_matrix, name_image):
    """
    Save the 2D binary image as a PNG file with 4xN pixels.
    :param image_matrix: the 2D binary image matrix
    :param name_image: the file name to save the image
    """
    img = Image.fromarray((image_matrix * 255).astype(np.uint8), 'L')
    img.save(name_image)



class TrainingDataGenerator:
    def __init__(self, path_csv, name_symbol, timeframe, num_candles, num_project_length, num_rows, num_MA,
                 flag_add_MA=False):

        # set up parameters
        self.datetime_format = '%Y-%m-%d %H:%M:%S+00:00'
        self.path_csv = path_csv
        self.name_symbol = name_symbol
        self.timeframe = timeframe
        self.num_candles = num_candles
        self.num_project_length = num_project_length
        self.num_rows = num_rows
        self.num_MA = num_MA
        self.df_OHLC = None
        self.df_dataset_log_training = None
        self.df_dataset_log_validation = None
        self.df_dataset_log_testing = None

        # Preprocessing the data
        self.read_and_preprocess_data()

        # save the data locally
        self.df_OHLC.to_csv('df_OHLC.csv')

    def read_and_preprocess_data(self):
        # Read in csv file
        df_OHLC_raw = pd.read_csv(self.path_csv, index_col=0)

        # Remove duplicates by keeping the first appearance
        self.df_OHLC = df_OHLC_raw.loc[~df_OHLC_raw.index.duplicated(keep='first')].copy()

        # Calculate the running high from last 1200 candles (1200/24 = 50 days)
        running_high_look_back_num = 1200
        self.df_OHLC['running_high'] = self.df_OHLC['High'].rolling(window=running_high_look_back_num).max()
        self.df_OHLC['running_low'] = self.df_OHLC['Low'].rolling(window=running_high_look_back_num).min()
        self.df_OHLC['running_hl_range'] = self.df_OHLC['running_high'] - self.df_OHLC['running_low']

        # Normalize the OHLC data by the running high and low
        self.df_OHLC['Open'] = (self.df_OHLC['Open'] - self.df_OHLC['running_low']) / self.df_OHLC['running_hl_range']
        self.df_OHLC['High'] = (self.df_OHLC['High'] - self.df_OHLC['running_low']) / self.df_OHLC['running_hl_range']
        self.df_OHLC['Low'] = (self.df_OHLC['Low'] - self.df_OHLC['running_low']) / self.df_OHLC['running_hl_range']
        self.df_OHLC['Close'] = (self.df_OHLC['Close'] - self.df_OHLC['running_low']) / self.df_OHLC['running_hl_range']

        ### Calculate the return and label
        self.df_OHLC['Close_future'] = self.df_OHLC['Close'].shift(-self.num_project_length)
        self.df_OHLC['return'] = (self.df_OHLC['Close_future'] - self.df_OHLC['Close']) / self.df_OHLC['Close']

        # Calculate the 30% and 70% percentile values of the return
        threshold_return_up = self.df_OHLC['return'].quantile(0.7)
        threshold_return_down = self.df_OHLC['return'].quantile(0.3)
        threshold_return_avg = (abs(threshold_return_up) + abs(threshold_return_down)) / 2
        threshold_return_up = threshold_return_avg

        # Generate the three class labels
        self.df_OHLC['label_three_class'] = (
            self.df_OHLC.apply(lambda row: 1 if row['return'] > threshold_return_up else -1 if row['return'] < threshold_return_down else 0, axis=1))

        # Generate the binary price trend label - up or down (>0 or <0)
        self.df_OHLC['label_trend_up'] = self.df_OHLC['return'].apply(lambda x: 1 if x > 0 else 0)

        ### Choose which label to use
        self.df_OHLC['label'] = self.df_OHLC['label_trend_up']

        ### Update the datetime start and end
        self.datetime_start_all = self.df_OHLC.index[0]
        self.datetime_end_all = self.df_OHLC.index[-1]
        self.df_OHLC.dropna(inplace=True)
        self.df_OHLC.to_csv('df_OHLC.csv')

    def generate_dataset(self, path_save, type_data, datetime_start, datetime_end, stride_size=None):
        """
        :param type_data: 'training', 'validation', or 'testing'
        :param datetime_start: the start datetime for the dataset
        :param datetime_end: the end datetime for the dataset
        :param interval: the interval between each session
        """
        dataset = self.df_OHLC[datetime_start:datetime_end]
        interval = stride_size if stride_size is not None else int(self.num_candles / 2)
        df_dataset_log = pd.DataFrame(columns=['id', 'datetime_end', 'label'])
        counter = 0

        if not os.path.exists(path_save):
            os.makedirs(path_save)

        for idx_end in range(self.num_candles, dataset.shape[0], interval):
            counter += 1
            idx_start = idx_end - self.num_candles
            dataset_session = dataset.iloc[idx_start:idx_end]
            label = dataset['label'].iloc[idx_end]
            name_file_prefix = f'{self.name_symbol}_{self.timeframe}_{counter:08d}_{label}'

            image_matrix = generate_2d_image(dataset_session)
            # name_image = f'{path_save}\\{name_file_prefix}.png'
            # save_image(image_matrix, name_image)
            name_image_npy = f'{path_save}\\{name_file_prefix}.npy'
            np.save(name_image_npy, image_matrix)

            df_dataset_log = pd.concat([df_dataset_log,
                                        pd.DataFrame({'id': [counter],
                                                      'datetime_end': [dataset.index[idx_end]],
                                                      'filename': name_file_prefix,
                                                      'label': [label]})])

        return df_dataset_log

    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_pickle(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


# 2024-04-27 study Generate the training set
if __name__ == '__main__':

    # asset parameters
    asset_type = "crypto"

    # image parameters
    num_candles = 60
    num_project_length = 3
    stride_size = 2
    num_rows = 60
    num_MA = 20

    # set up paths
    path_data = 'C:\\Data'

    # list of name symbols
    # name_symbol_list = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SOLUSDT', 'NEARUSDT', 'LINKUSDT']
    # name_symbol_list = ['BTCUSDT', 'SOLUSDT']
    name_symbol_list = ['BTCUSDT']

    # list of timeframes
    # timeframe_list = ['1h', '4h', '12h', '1d']
    # timeframe_list = ['1h', '4h']
    timeframe_list = ['1h']

    # list of data types
    data_type_list = ['training', 'validation', 'testing']

    for name_symbol in name_symbol_list:
        for timeframe in timeframe_list:

            # create the generator
            path_csv = f'data\\{asset_type}\\{name_symbol}_{timeframe}.csv'
            generator = TrainingDataGenerator(path_csv=path_csv,
                                              name_symbol=name_symbol,
                                              timeframe=timeframe,
                                              num_candles=num_candles,
                                              num_project_length=num_project_length,
                                              num_rows=num_rows,
                                              num_MA=num_MA)

            # save the object locally for later use to generate testing data
            datetime_now = datetime.datetime.now().strftime('%Y%m%d')
            path_save_obj = f'{path_data}\\generator_{name_symbol}_{timeframe}_{datetime_now}.pkl'
            generator.save_to_pickle(path_save_obj)

            # specify the datetime_start and datetime_end for the training, validation, and testing data using a dict
            datetime_start_dict = {'training': generator.df_OHLC.index[60],
                                   'validation': '2022-04-01 00:00:00+00:00',
                                   'testing': '2023-04-01 00:00:00+00:00'}

            datetime_end_dict = {'training': '2022-03-01 00:00:00+00:00',
                                 'validation': '2023-03-01 00:00:00+00:00',
                                 'testing': '2024-04-15 00:00:00+00:00'}

            # Now generate the training, validation, and testing data
            for type_data in data_type_list:
                print(f'Processing {name_symbol} {timeframe} {type_data}...')

                # Load the generator, redundantly, but just to make sure the saved the generator is working
                generator = TrainingDataGenerator.load_from_pickle(path_save_obj)

                # Get the datetime_start and datetime_end
                datetime_start = datetime_start_dict[type_data]
                datetime_end = datetime_end_dict[type_data]

                # Generate the dataset
                path_save = f'{path_data}\\data_{type_data}_{name_symbol}_{timeframe}'
                df_dataset_log = generator.generate_dataset(path_save=path_save,
                                                            type_data=type_data,
                                                            datetime_start=datetime_start,
                                                            datetime_end=datetime_end,
                                                            stride_size=stride_size)

                # save the df_OHLC temporarily
                generator.df_OHLC.to_csv('temp.csv')

                # save the log
                df_dataset_log.to_csv(f'{path_data}\dataset_log_{name_symbol}_{timeframe}_{type_data}.csv')





    
    # path_save_training = f'{path_save}\\data_training_{name_symbol}_{timeframe}'
    # df_dataset_log_training = generator.generate_dataset(type_data='training',
    #                                                      path_save=path_save_training,
    #                                                      datetime_start=datetime_start_training,
    #                                                      datetime_end=datetime_end_training,
    #                                                      stride_size=stride_size)
    # df_dataset_log_training.to_csv(f'dataset_log_{name_symbol}_{timeframe}_training.csv')
    # 
    # # Generate the validation set
    # datetime_start_validation = '2022-04-01 00:00:00+00:00'
    # datetime_end_validation = '2023-02-01 00:00:00+00:00'
    # path_save_validation = f'{path_save}\\data_validation_{name_symbol}_{timeframe}'
    # df_dataset_log_validation = generator.generate_dataset(type_data='validation',
    #                                                        path_save=path_save_validation,
    #                                                        datetime_start=datetime_start_validation,
    #                                                        datetime_end=datetime_end_validation,
    #                                                        stride_size=stride_size)
    # df_dataset_log_validation.to_csv(f'dataset_log_{name_symbol}_{timeframe}_validation.csv')
    # 
    # # Generate the testing set
    # datetime_start_testing = '2023-03-01 00:00:00+00:00'
    # datetime_end_testing = '2024-04-01 00:00:00+00:00'
    # path_save_testing = f'{path_save}\\data_testing_{name_symbol}_{timeframe}'
    # df_dataset_log_testing = generator.generate_dataset(type_data='testing',
    #                                                     path_save=path_save_testing,
    #                                                     datetime_start=datetime_start_testing,
    #                                                     datetime_end=datetime_end_testing,
    #                                                     stride_size=stride_size)
    # df_dataset_log_testing.to_csv(f'dataset_log_{name_symbol}_{timeframe}_testing.csv')



    # # specify the datetime_start and datetime_end for the training, validation, and testing data using a dict
    # datetime_start_dict = {'training': generator.df_OHLC.index[60],
    #                        'validation': '2023-07-01 00:00:00+00:00',
    #                        'testing': '2024-01-01 00:00:00+00:00'}
    #
    # datetime_end_dict = {'training': '2023-05-31 00:00:00+00:00',
    #                      'validation': '2023-11-30 00:00:00+00:00',
    #                      'testing': '2024-04-15 00:00:00+00:00'}