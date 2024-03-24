from sys import path
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import boxcox1p
import seaborn as sns
from tensorflow.keras.models import Sequential,load_model
from scipy import stats
from scipy.stats import norm, skew 
from tensorflow.keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout,Bidirectional,GRU
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from os import path

plt.rcParams["figure.figsize"] = (7,5)

import warnings
warnings.filterwarnings('ignore')

dir = path.dirname(__file__)

def predicted_ftmscan(data):
    
    list_append = list()
    
    data = pd.DataFrame(data).transpose()
    list_append.append(data)
    data.columns = ['pendingcount','avgtxnsperblock']
    data['avgtxnsperblock'] = boxcox1p(data['avgtxnsperblock'] , 0.15)
    data['pendingcount'] = boxcox1p(data['pendingcount'] , 0.15)
    data = np.asarray(data)
    
    rapid_gas_scaler = joblib.load(path.join(dir, './data_ftmscan_std_scaler.bin'))
    rapid_gas_scaler = rapid_gas_scaler.transform(data)
    rapid_gas_scaler = np.reshape(rapid_gas_scaler, (rapid_gas_scaler.shape[0], 1, rapid_gas_scaler.shape[1]))
    
    rapid_gas_lstm = load_model(path.join(dir, './Model_weights_ftmscan/data_ftmscan_rapid_lstm.h5'))
    rapid_gas_bilstm = load_model(path.join(dir, './Model_weights_ftmscan/data_ftmscan_rapid_bilstm.h5'))
    rapid_gas_gru = load_model(path.join(dir, './Model_weights_ftmscan/data_ftmscan_rapid_rapid_gru.h5'))
    
    rapid_gas_lstm = rapid_gas_lstm.predict(rapid_gas_scaler)
    rapid_gas_bilstm = rapid_gas_bilstm.predict(rapid_gas_scaler)
    rapid_gas_gru = rapid_gas_gru.predict(rapid_gas_scaler)
    
    
    rapid_gas_lstm_error = load_model(path.join(dir, './Model_weights_ftmscan/data_ftmscan_rapid_error_lstm.h5'))
    rapid_gas_gru_error = load_model(path.join(dir, './Model_weights_ftmscan/data_ftmscan_rapid_error_gru.h5'))
    rapid_gas_bilstm_error = load_model(path.join(dir, './Model_weights_ftmscan/data_ftmscan_rapid_error_bilstm.h5'))
    
    rapid_gas_lstm_error = np.argmax(rapid_gas_lstm_error.predict(rapid_gas_scaler))
    rapid_gas_bilstm_error = np.argmax(rapid_gas_bilstm_error.predict(rapid_gas_scaler))
    rapid_gas_gru_error = np.argmax(rapid_gas_gru_error.predict(rapid_gas_scaler))
    
    print('LSTM RapidGas ', rapid_gas_lstm)
    print('BiLSTM RapidGas ', rapid_gas_bilstm)
    print('GRU Rapid Gas', rapid_gas_gru)
    
    print('*'* 30)
    
    print('LSTM Error Class ', rapid_gas_lstm_error)
    print('BiLSTM Error Class ', rapid_gas_bilstm_error)
    print('GRU Error Class', rapid_gas_gru_error)
    
    if rapid_gas_lstm_error == 0:
        print('Error is zero of LSTM: ', rapid_gas_lstm)
        
    elif rapid_gas_bilstm_error == 0:
        print('Error is zero of BiLSTM ' , rapid_gas_bilstm)
        
    else:
        print('Error is zero of GRU ',rapid_gas_gru)
        
    rapid_gas_lstm = rapid_gas_lstm.tolist()    
    rapid_gas_bilstm = rapid_gas_bilstm.tolist()
    rapid_gas_gru = rapid_gas_gru.tolist()
    
    list_append.append(rapid_gas_lstm)
    list_append.append(rapid_gas_bilstm)
    list_append.append(rapid_gas_gru)
    
    rapid_gas_lstm_error = rapid_gas_lstm_error.tolist()
    rapid_gas_bilstm_error = rapid_gas_bilstm_error.tolist()
    rapid_gas_gru_error = rapid_gas_gru_error.tolist()
    
    list_append.append(rapid_gas_lstm_error)
    list_append.append(rapid_gas_bilstm_error)
    list_append.append(rapid_gas_gru_error)
    
    final_frame = pd.DataFrame(list_append).transpose()
    final_frame = final_frame.iloc[:,1:]
    final_frame.columns = ['Gas LSTM','Gas BiLstm','Gas GRU','Error LSTM','Error BiLstm','Error GRU']
    final_frame['Gas LSTM'] = final_frame['Gas LSTM'].str.get(0)
    final_frame['Gas BiLstm'] = final_frame['Gas BiLstm'].str.get(0)
    final_frame['Gas GRU'] = final_frame['Gas GRU'].str.get(0)
    final_frame['Gas LSTM'] = final_frame['Gas LSTM'].str.get(0)
    final_frame['Gas BiLstm'] = final_frame['Gas BiLstm'].str.get(0)
    final_frame['Gas GRU'] = final_frame['Gas GRU'].str.get(0)
    
    final_frame.to_csv('code_ftmscan.csv',index=None)

data = [8000,54]

if __name__ == "__main__":
    predicted_ftmscan(map(float, sys.argv[1:]))
        

    