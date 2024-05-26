import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pickle

class LSTMDataGenerator:
    def __init__(self, path_csv, num_candles, num_project_length):
        self.path_csv = path_csv
        self.num_candles = num_candles
        self.num_project_length = num_project_length
        self.df_OHLC = None
        self.scaler = None  # Will be initialized after feature selection

        self.read_and_preprocess_data()

    def read_and_preprocess_data(self):
        df_OHLC_raw = pd.read_csv(self.path_csv, index_col=0)
        df_OHLC_raw.index = pd.to_datetime(df_OHLC_raw.index)
        df_OHLC_raw = df_OHLC_raw.loc[~df_OHLC_raw.index.duplicated(keep='first')]

        # Calculate future return
        df_OHLC_raw['Close_future'] = df_OHLC_raw['Close'].shift(-self.num_project_length)
        df_OHLC_raw['return'] = (df_OHLC_raw['Close_future'] - df_OHLC_raw['Close']) / df_OHLC_raw['Close']

        # Calculate the top 30% of return as 1, and the rest as 0
        df_OHLC_raw['label'] = 0
        df_OHLC_raw.loc[df_OHLC_raw['return'] > df_OHLC_raw['return'].quantile(0.55), 'label'] = 1

        # Calculate technical indicators
        df_OHLC_raw['RSI'] = talib.RSI(df_OHLC_raw['Close'].values, timeperiod=14)
        df_OHLC_raw['20EMA'] = talib.EMA(df_OHLC_raw['Close'].values, timeperiod=20)
        df_OHLC_raw['50EMA'] = talib.EMA(df_OHLC_raw['Close'].values, timeperiod=50)
        df_OHLC_raw['100EMA'] = talib.EMA(df_OHLC_raw['Close'].values, timeperiod=100)

        # Calculate Bollinger Bands
        df_OHLC_raw['upper_band'], df_OHLC_raw['middle_band'], df_OHLC_raw['lower_band'] = talib.BBANDS(df_OHLC_raw['Close'].values, timeperiod=20)
        df_OHLC_raw['bollinger_coeff'] = (df_OHLC_raw['Close'] - df_OHLC_raw['middle_band']) / (df_OHLC_raw['upper_band'] - df_OHLC_raw['lower_band'])

        # Calculate the normalizer for price and volume
        running_high_look_back_num = 1200
        df_OHLC_raw['running_high'] = df_OHLC_raw['High'].rolling(window=running_high_look_back_num).max()
        df_OHLC_raw['running_low'] = df_OHLC_raw['Low'].rolling(window=running_high_look_back_num).min()
        df_OHLC_raw['running_hl_range'] = df_OHLC_raw['running_high'] - df_OHLC_raw['running_low']
        df_OHLC_raw['Volume_high'] = df_OHLC_raw['Volume'].rolling(window=running_high_look_back_num).max()
        df_OHLC_raw['Volume_low'] = df_OHLC_raw['Volume'].rolling(window=running_high_look_back_num).min()
        df_OHLC_raw['Volume_range'] = df_OHLC_raw['Volume_high'] - df_OHLC_raw['Volume_low']

        # Normalize features
        df_OHLC_raw['Close'] = (df_OHLC_raw['Close'] - df_OHLC_raw['running_low']) / df_OHLC_raw['running_hl_range']
        df_OHLC_raw['deviation_20EMA'] = (df_OHLC_raw['Close'] - df_OHLC_raw['20EMA']) / df_OHLC_raw['running_hl_range']
        df_OHLC_raw['deviation_50EMA'] = (df_OHLC_raw['Close'] - df_OHLC_raw['50EMA']) / df_OHLC_raw['running_hl_range']
        df_OHLC_raw['deviation_100EMA'] = (df_OHLC_raw['Close'] - df_OHLC_raw['100EMA']) / df_OHLC_raw['running_hl_range']
        df_OHLC_raw['RSI'] = df_OHLC_raw['RSI'] / 100
        df_OHLC_raw['Volume'] = (df_OHLC_raw['Volume'] - df_OHLC_raw['Volume_low']) / df_OHLC_raw['Volume_range']

        df_OHLC_raw.dropna(inplace=True)
        self.df_OHLC = df_OHLC_raw

        # Select features for training
        self.features = ['Close', 'deviation_20EMA', 'deviation_50EMA', 'deviation_100EMA', 'RSI', 'bollinger_coeff', 'Volume']
        self.df_OHLC = self.df_OHLC[['label'] + self.features]

    def fit_scaler(self, selected_features):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.scaler.fit(self.df_OHLC[selected_features])

    def generate_sequences(self, start_idx, end_idx, selected_features=None):
        sequences = []
        labels = []
        for i in range(start_idx, end_idx):
            end_pos = i + self.num_candles
            if end_pos + self.num_project_length > len(self.df_OHLC):
                break
            seq = self.df_OHLC.iloc[i:end_pos].drop(columns=['label'])
            if selected_features is not None:
                seq = seq[selected_features]
            scaled_seq = self.scaler.transform(seq)
            sequences.append(scaled_seq)
            labels.append(self.df_OHLC['label'].iloc[end_pos + self.num_project_length - 1])
        return np.array(sequences), np.array(labels)

    def save_to_pickle(self, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_from_pickle(file_path):
        with open(file_path, 'rb') as file:
            return pickle.load(file)


if __name__ == '__main__':
    path_csv = 'data\\crypto\\BTCUSDT_1h.csv'
    num_candles = 60  # Use 60 time steps for consistency
    num_project_length = 5

    generator = LSTMDataGenerator(path_csv, num_candles, num_project_length)

    len_train = int(len(generator.df_OHLC) * 0.7)
    len_val = int(len(generator.df_OHLC) * 0.85)
    len_test = len(generator.df_OHLC)

    # Use RandomForest to determine feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(generator.df_OHLC[generator.features], generator.df_OHLC['label'])
    feature_importances = pd.Series(rf.feature_importances_, index=generator.features)
    selected_features = feature_importances.nlargest(5).index.tolist()
    print("Selected features:", selected_features)

    # Fit the scaler with the selected features
    generator.fit_scaler(selected_features)

    train_sequences, train_labels = generator.generate_sequences(0, len_train, selected_features=selected_features)
    val_sequences, val_labels = generator.generate_sequences(len_train, len_val, selected_features=selected_features)
    test_sequences, test_labels = generator.generate_sequences(len_val, len_test, selected_features=selected_features)

    # Calculate class weights to handle class imbalance
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = {0: class_weights[0], 1: class_weights[1] * 1.2}  # Increase weight for the minority class

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(num_candles, train_sequences.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(train_sequences, train_labels, epochs=100, batch_size=32, validation_data=(val_sequences, val_labels), callbacks=[early_stop], class_weight=class_weights)

    loss, accuracy = model.evaluate(test_sequences, test_labels)
    print(f'Test Accuracy: {accuracy:.4f}')

    # Predictions
    test_predictions = model.predict(test_sequences)
    test_predictions = (test_predictions > 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_predictions)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Positive', 'Positive'])
    cm_display.plot(cmap=plt.cm.Blues)
    plt.show()

    # Classification Report
    print(classification_report(test_labels, test_predictions, target_names=['Non-Positive', 'Positive']))
