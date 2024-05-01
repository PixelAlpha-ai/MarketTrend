""" Define a function to read a pandas dataframe of OHLCV data and to convert into 2D binary image"""
import pandas as pd
import datetime
import numpy as np
import talib
import os
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"


def generate_2d_image(df_OHLC_session, num_rows, flag_add_MA=False, flag_show_OHLC=False):

    """
    :param df_OHLC_session: the session data that contains OHLCV, and the running min/max for the plot range
    :return: the 2D binary image of the OHLC data, and the OHLC plot of the same image
    """

    # Get the running min and max
    running_high = df_OHLC_session['running_high'].iloc[-1]
    running_low = df_OHLC_session['running_low'].iloc[-1]

    if flag_show_OHLC:
        import plotly.graph_objects as go
        # generate a candlestick image of the OHLC data with the MA using plotly library
        fig = go.Figure(data=[go.Candlestick(x=df_OHLC_session.index,
                                             open=df_OHLC_session['Open'],
                                             high=df_OHLC_session['High'],
                                             low=df_OHLC_session['Low'],
                                             close=df_OHLC_session['Close'])])

        # remove the slider bar
        fig.update_xaxes(rangeslider_visible=False)

        # add the MA
        if flag_add_MA:
            import plotly.graph_objects as go
            fig.add_trace(go.Scatter(x=df_OHLC_session.index, y=df_OHLC_session['MA'], line=dict(color='blue', width=1)))

        # set the min and max to be the current running min and max, which is the last value in the dataframe
        fig.update_yaxes(range=[running_low, running_high])

        # # save the figure as a png file
        fig_OHLC = fig

    else:
        fig_OHLC = None

    # Generate the 2D binary image with bamboo candlesticks
    # Create a 2D numpy array to store the image
    # num_cols = 3 x N, where N is the number of candles
    # num_rows = user defined.
    num_cols = df_OHLC_session.shape[0] * 3
    image_binary = np.zeros((num_rows, num_cols))

    # Fill in the image with the OHLC data
    for i in range(df_OHLC_session.shape[0]):

        # Get the OHLC data
        open = df_OHLC_session['Open'].iloc[i]
        high = df_OHLC_session['High'].iloc[i]
        low = df_OHLC_session['Low'].iloc[i]
        close = df_OHLC_session['Close'].iloc[i]

        # Get the MA
        if flag_add_MA:
            MA = df_OHLC_session['MA'].iloc[i]

        # Calculate the candlestick position
        col = i * 3
        col_open = col
        col_high_low = col + 1
        col_close = col + 2

        # calculate the row interval size
        row_interval = (running_high - running_low) / num_rows

        # Calculate the candlestick height, the height should be the whole integer value after the remainder operation
        row_high = int((running_high - high) / row_interval)
        row_low = int((running_high - low) / row_interval)
        row_open = int((running_high - open) / row_interval)
        row_close = int((running_high - close) / row_interval)

        # flip the values, the previous values ranged from 0 to 29, so now it should be from 29 to 0
        row_high = num_rows - row_high - 1
        row_low = num_rows - row_low - 1
        row_open = num_rows - row_open - 1
        row_close = num_rows - row_close - 1

        # Fill in the candlestick
        image_binary[row_low:row_high, col_high_low] = 1
        image_binary[row_open, col_open] = 1
        image_binary[row_close, col_close] = 1

    return image_binary, fig_OHLC

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

        # flags
        self.flag_add_MA = flag_add_MA

        # Preprocessing the data
        self.read_and_preprocess_data()

        # save the data locally
        self.df_OHLC.to_csv('df_OHLC.csv')


    def read_and_preprocess_data(self):

        # Read in csv file
        df_OHLC_raw = pd.read_csv(self.path_csv, index_col=0)

        # Remove duplicates by keeping the first appearance
        self.df_OHLC = df_OHLC_raw.loc[~df_OHLC_raw.index.duplicated(keep='first')].copy()

        ### Find the running min and max for the 2D image y-xis range
        # Calculate the ATR and its cutoff
        self.df_OHLC['ATR'] = talib.ATR(self.df_OHLC['High'], self.df_OHLC['Low'], self.df_OHLC['Close'],
                                        timeperiod=self.num_candles)
        self.ATR_cutoff = self.df_OHLC['ATR'].quantile(0.8)

        # Calculate the running high and low for the last num_candle OHLC data
        self.df_OHLC['running_high'] = self.df_OHLC['High'].rolling(window=self.num_candles).max()
        self.df_OHLC['running_low'] = self.df_OHLC['Low'].rolling(window=self.num_candles).min()
        self.df_OHLC['running_mid'] = (self.df_OHLC['running_high'] + self.df_OHLC['running_low']) / 2

        # if MA should be considered
        if self.flag_add_MA:
            # Calculate the MA
            self.df_OHLC['MA'] = talib.MA(self.df_OHLC['Close'], timeperiod=self.num_MA)

            # Calculate the running high and low for the last num_candle MA data
            self.df_OHLC['running_high_MA'] = self.df_OHLC['MA'].rolling(window=self.num_MA).max()
            self.df_OHLC['running_low_MA'] = self.df_OHLC['MA'].rolling(window=self.num_MA).min()

            # Keep the higher and lower values between the running high of MA and OHLC
            self.df_OHLC['running_high'] = self.df_OHLC[['running_high_MA', 'running_high']].max(axis=1)
            self.df_OHLC['running_low'] = self.df_OHLC[['running_low_MA', 'running_low']].min(axis=1)
            self.df_OHLC['running_mid'] = (self.df_OHLC['running_high'] + self.df_OHLC['running_low']) / 2

        # Get the running range for the running highs and lows
        self.df_OHLC['running_range'] = self.df_OHLC['running_high'] - self.df_OHLC['running_low']

        # Check if any of the running range is less than the ATR cutoff, if yes, then expand the range to fit the ATR cutoff
        self.df_OHLC['running_range'] = self.df_OHLC['running_range'].apply(
            lambda x: x if x > self.ATR_cutoff else self.ATR_cutoff)

        # Re-calculate the running high and low (after considering the ATR), as the true range for plots
        self.df_OHLC['running_high'] = self.df_OHLC['running_mid'] + self.df_OHLC['running_range'] / 2
        self.df_OHLC['running_low'] = self.df_OHLC['running_mid'] - self.df_OHLC['running_range'] / 2

        ### Calculate the return and label
        # Calculate the return (calculated as the change after num_project_length candles)
        # 1 - define a new array to store the close price after num_project_length candles
        self.df_OHLC['Close_future'] = self.df_OHLC['Close'].shift(-self.num_project_length)
        self.df_OHLC['return'] = (self.df_OHLC['Close_future'] - self.df_OHLC['Close']) / self.df_OHLC['Close']

        # debug
        # self.df_OHLC.to_csv('df_OHLC.csv')

        # Calculate the top and bottom 5% VaR using entire dataset's return
        VaR_top_all = self.df_OHLC['return'].quantile(0.95)
        VaR_bottom_all = self.df_OHLC['return'].quantile(0.05)
        print(f'VaR_top_all: {VaR_top_all}, VaR_bottom_all: {VaR_bottom_all}')

        # Manually define 5% as the top and bottom VaR
        # VaR_top_all = 0.05
        # VaR_bottom_all = -0.05

        # generate the label using the top and bottom 5% VaR
        self.df_OHLC['label_down_var'] = (
            self.df_OHLC.apply(lambda row: 1 if row['return'] < VaR_bottom_all else 0, axis=1))
        self.df_OHLC['label_up_var'] = (
            self.df_OHLC.apply(lambda row: 1 if row['return'] > VaR_top_all else 0, axis=1))

        # generate the label using simply the trend (up for 1; down for 0)
        self.df_OHLC['label_trend_up'] = (
            self.df_OHLC.apply(lambda row: 1 if row['return'] > 0. else 0, axis=1))

        # Calculate the top and bottom 5% VaR using rolling recent 1000 candles' return
        # self.df_OHLC['VaR_top'] = self.df_OHLC['return'].rolling(window=1000).quantile(0.95)
        # self.df_OHLC['VaR_bottom'] = self.df_OHLC['return'].rolling(window=1000).quantile(0.05)
        # self.df_OHLC['label_down_var'] = (
        #     self.df_OHLC.apply(lambda row: 1 if row['return'] < row['VaR_bottom'] else 0, axis=1))
        # self.df_OHLC['label_up_var'] = (
        #     self.df_OHLC.apply(lambda row: 1 if row['return'] > row['VaR_top'] else 0, axis=1))
        # self.df_OHLC.dropna(inplace=True)

        ### Choose which label to use
        # self.df_OHLC['label'] = self.df_OHLC['label_down_var']
        self.df_OHLC['label'] = self.df_OHLC['label_trend_up']

        ### Update the datetime start and end
        self.df_OHLC.dropna(inplace=True)
        self.datetime_start_all = self.df_OHLC.index[0]
        self.datetime_end_all = self.df_OHLC.index[-1]

        # print a report of the data summary, including length and etc


        ### Visualization of the label using plotly.
        # # create a figure
        # fig = go.Figure()
        #
        # # get a random sample of the data with 1% of the data
        # df_OHLC_sample = self.df_OHLC.sample(frac=0.01)
        #
        # # add the returns as bars
        # df_OHLC_sample.to_csv('df_OHLC_sample.csv')
        # fig.add_trace(go.Bar(x=df_OHLC_sample.index, y=df_OHLC_sample['return'], name='return'))
        #
        #
        # # add the labels as verticle spikes on the plot
        # fig.add_trace(go.Scatter(x=df_OHLC_sample.index, y=df_OHLC_sample['return'] * df_OHLC_sample['label_down_var'],
        #                             mode='markers', marker=dict(color='red'), name='label_down_var'))
        #
        # # set the figure show to be on browser
        # fig.show()


    def generate_example_image(self, datetime_end, flag_show_OHLC=True):
        """ Generate just one binary and typical OHLC image for visual"""

        # Calculate time_end, which is the time_start plus num_candles
        datetime_end_dt = pd.to_datetime(datetime_end)
        datetime_start_dt = datetime_end_dt - datetime.timedelta(hours=self.num_candles-1)
        datetime_start = datetime_start_dt.strftime(self.datetime_format)

        # Get the data session
        df_OHLC_session = self.df_OHLC[datetime_start:datetime_end]

        # Generate the 2D binary image
        image_binary, fig_OHLC = generate_2d_image(df_OHLC_session,
                                                   num_rows=self.num_rows,
                                                   flag_show_OHLC=flag_show_OHLC)

        return image_binary, fig_OHLC


    def generate_dataset(self, path_save, type_data, datetime_start, datetime_end, stride_size=None):
        """ generate the training, validation, and testing dataset using the given time period
        :param type_data: 'training', 'validation', or 'testing'
        :param datetime_start: the start datetime for the dataset
        :param datetime_end: the end datetime for the dataset
        :param interval: the interval between each session
        """

        # Get the data section
        dataset = self.df_OHLC[datetime_start:datetime_end]

        # Set the interval
        if stride_size is None:
            interval = int(self.num_candles/2)
        else:
            interval = stride_size

        # Initialize a dataframe for logging
        df_dataset_log = pd.DataFrame(columns=['id', 'datetime_end', 'label'])
        counter = 0

        # Check if the folder exists, if not, create the folder
        path_save_folder = path_save
        if not os.path.exists(path_save_folder):
            os.makedirs(path_save_folder)

        for idx_end in range(self.num_candles, dataset.shape[0], interval):

            counter += 1

            # get the start and end index
            idx_start = idx_end - self.num_candles
            dataset_session = dataset.iloc[idx_start:idx_end]

            # get the label
            label = dataset['label'].iloc[idx_end]

            # get the file prefix
            name_file_prefix = f'{self.name_symbol}_{self.timeframe}_{counter:08d}_{label}'
            # print(f'Processing {name_file_prefix}...')

            # generate and save the 2D image
            image_binary, _ = generate_2d_image(dataset_session, num_rows=self.num_rows)
            # name_image = f'data_{type_data}\\{name_file_prefix}.png'
            # plt.imsave(name_image, image_binary, cmap='gray', format='png')
            name_image_npy = f'{path_save}\\{name_file_prefix}.npy'
            np.save(name_image_npy, image_binary)

            # update the log
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
    num_project_length = 5
    stride_size = 2
    num_rows = 60
    num_MA = 20

    # set up paths
    path_data = 'C:\\Data'

    # list of name symbols
    name_symbol_list = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'DOGEUSDT', 'SOLUSDT', 'NEARUSDT', 'LINKUSDT']
    # name_symbol_list = ['BTCUSDT', 'SOLUSDT']
    # name_symbol_list = ['BTCUSDT']

    # list of timeframes
    timeframe_list = ['1h', '4h', '12h', '1d']
    timeframe_list = ['1h', '4h']
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

            # Now gereate the training, validation, and testing data
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