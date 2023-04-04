import numpy as np
import os
from datetime import datetime, timezone, timedelta
import random
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
NUM_HISTORICAL_FEATURES = 1
TOTAL_FLOW_FEATURE_INDEX = 0




# TRAINING DATA PARAMETERS
TRAINING_TYPE = 'train'
TRAINING_YEAR = 2023
TRAINING_DISTRICT = 10
NUM_TRAINING_STATIONS = 3
TRAINING_DATETIME_BEGIN = datetime(year=TRAINING_YEAR, month=1, day=1, hour=0, minute=0)
TRAINING_DATETIME_END = datetime(year=TRAINING_YEAR, month=1, day=5, hour=0, minute=0)

# Input WINDOWS PARAMETERS
SAMPLE_PERIOD = timedelta(minutes=30) # How much time per data sampled
HISTORICAL_WINDOW = timedelta(days=3, hours=0, minutes=0)
FORECAST_WINDOW = timedelta(days=0, hours=12, minutes=0)
STRIDE_WINDOW = timedelta(days=0, hours=12, minutes=0)

#path name
#TRAINING_DIRECTORY = r'C:/Users/poonammetkar/Documents/one_shot_time_series_forecast_tinker/preprocessed_data/district_3/year_2023/pickle'
TRAINING_DIRECTORY = r'C:/Users/poonammetkar/Documents/one_shot_time_series_forecast_tinker/preprocessed_data/district_10/year_2023/pickle'
#TRAINING_DIRECTORY = r'G:\one_shot_time_series_forecast_tinker\preprocessed_data\district_{}\year_{}\pickle'.format(TRAINING_DISTRICT, TRAINING_YEAR) # These would hold pickle files created from .gz files using the PeMS.py class

DATA_DIR =  r'C:/Users/poonammetkar/Documents/dataset/district_10'

# ----------------------[CONSTANTS]----------------------
# Converting our windows to number of samples
NUM_HISTORICAL_SAMPLES = int(HISTORICAL_WINDOW/SAMPLE_PERIOD)
NUM_FORECAST_SAMPLES = int(FORECAST_WINDOW/SAMPLE_PERIOD)
NUM_STRIDE_SAMPLES = int(STRIDE_WINDOW/SAMPLE_PERIOD)
# Random seed to make "randomization" repeatable as long as the seed is the same
RANDOM_SEED = 8675309
# The 1 forecast feature is Traffic Total Flow; We are not focusing on predicting anything else
NUM_FORECAST_FEATURES = 1

def window_data(data, w=NUM_HISTORICAL_SAMPLES+NUM_FORECAST_SAMPLES, f=NUM_HISTORICAL_FEATURES, stride=NUM_STRIDE_SAMPLES):
  ''' Rolling window on the data, truncate end of data that does not have enough data for the next/last stride '''
  return np.lib.stride_tricks.sliding_window_view(x=data, window_shape=(w, f))[::stride].reshape(-1, w, f)

def create_data(directory, number_stations, datetime_begin, datetime_end, stride):
  candidate_files = []
  for filename in os.listdir(directory):
    candidate_files.append(filename)
    candidate_files = random.sample(candidate_files, min(len(candidate_files), number_stations))
    historical= None
    forecast = None

  historical_windows = []
  forecast_windows = []

  for index, filename in enumerate(candidate_files):
    print('INDEX: {}'.format(index))
    #out = []
    #print('GENERATING PAIRWISE DATA FOR STATION: {}'.format(filename))
    file = os.path.join(directory, filename)
    if os.path.isfile(file):
      print(file)
      # Read the pickle file to get the sample_station_df (remember, this data is the same, data we graphed)
      sample_station_df = pd.read_pickle(file)
      # Resample the sample_station_df (sampled originally at 5 minutes) to a new USER_CONFIGURATION sample period
      sample_station_df = sample_station_df.resample(SAMPLE_PERIOD).sum()
     
      # In order to optimize speed, work with purely numpy arrays when generating the pairwise data
      station_data = sample_station_df.to_numpy()
      window_size = NUM_HISTORICAL_SAMPLES+NUM_FORECAST_SAMPLES

      shape = ((station_data.shape[0]-window_size)//NUM_STRIDE_SAMPLES)+1,window_size
      strides = (station_data.strides[0] * NUM_STRIDE_SAMPLES, station_data.strides[0])

      window_view = np.lib.stride_tricks.as_strided(station_data, shape=shape, strides=strides)

      data_mean = np.mean(window_view)
      data_std = np.std(window_view)

      #1. normalization method 2. min-max scaling 3. log transformation
      #z score normalization method  
      normalized_data = (window_view - data_mean) / data_std


      station_windowed_historical_data = window_view[:,:NUM_HISTORICAL_SAMPLES]
      station_windowed_forecast_data = window_view[:,NUM_HISTORICAL_SAMPLES:]

      # Add the historical and forecast windows for this station into a conglomerated list of windows for all stations
      historical_windows.append(station_windowed_historical_data)
      forecast_windows.append(station_windowed_forecast_data)

      station_windowed_historical_data.shape
      num_historical_samples = NUM_HISTORICAL_SAMPLES
      num_forecast_samples = NUM_FORECAST_SAMPLES

      np.savez_compressed(os.path.join(DATA_DIR, filename),
                        historical=historical,
                        forecast=forecast)
      
      x_historical = np.arange(num_historical_samples)
      x_forecast = np.arange(num_historical_samples, num_historical_samples + num_forecast_samples)
      plt.plot(x_historical, station_windowed_historical_data[2], color='purple', label='historical')
      plt.plot(x_forecast, station_windowed_forecast_data[2], color='orange', label='forecast')
      plt.title('Time Series Data')
      plt.xlabel('time series points = 30 min')
      plt.ylabel('traffic flow')
      plt.savefig('timeseries.png')
      plt.xlim(0,168)
      plt.show()
      plt.close()
      

      station_normalize_historical_data = normalized_data[:,:NUM_HISTORICAL_SAMPLES]
      station_normalize_forecast_data = normalized_data[:,NUM_HISTORICAL_SAMPLES:]

      plt.plot(x_historical, station_normalize_historical_data[2], color='purple', label='historical')
      plt.plot(x_forecast, station_normalize_forecast_data[2], color='orange', label='forecast')
      plt.title('Normalized Time Series Data')
      plt.xlabel('time series points = 30 min')
      plt.ylabel('traffic flow')
      plt.savefig('normalize_timeseries.png')
      plt.xlim(0,168)
      plt.show()
      plt.close()
      

       



  # When all is said and done, there should be at least one station data file that has valid data and has successfully extracted its rolling window data
  assert(len(historical_windows) > 0)
  assert(len(forecast_windows) > 0)

  # Transform a list of numpy arrays to a numpy-list of numpy arrays
  historical_windows_np = np.concatenate(historical_windows, axis=0)
  forecast_windows_np = np.concatenate(forecast_windows, axis=0)

  historical, forecast = shuffle(historical_windows_np, forecast_windows_np, random_state=RANDOM_SEED)
  return historical, forecast

  '''
  def plot_windowed_data(station_windowed_data,
                       num_historical_samples=NUM_HISTORICAL_SAMPLES, num_forecast_samples=NUM_FORECAST_SAMPLES,
                       num_windows_to_plot=3):
    #Plot the windowed data
    window_start_index = min(0, station_windowed_data.shape[0]) # Arbitrary indices chosen
    window_end_index = min(num_windows_to_plot, station_windowed_data.shape[0])
    for i in range(window_start_index, window_end_index):
        sample_historical = np.copy(station_windowed_data[i, :, TOTAL_FLOW_FEATURE_INDEX])
        sample_historical = sample_historical[:num_historical_samples]

        sample_forecast = np.copy(station_windowed_data[i, :, TOTAL_FLOW_FEATURE_INDEX])
        sample_forecast = sample_forecast[num_historical_samples:]

        plt.title("Example window: {}".format(i))
        plt.xlabel("Period")
        plt.ylabel("Total Flow")

        x_historical = np.arange(num_historical_samples)
        x_forecast = np.arange(num_historical_samples, num_historical_samples + num_forecast_samples)
        plt.plot(x_historical, sample_historical, color='purple', label='historical')
        plt.plot(x_forecast, sample_forecast, color='orange', label='forecast')
        plt.show()
'''

def save_data(filename, historical, forecast):
    # https://stackoverflow.com/questions/35133317/numpy-save-some-arrays-at-once
    np.savez_compressed(os.path.join(DATA_DIR, filename),
                        historical=historical,
                        forecast=forecast)
    print("save file")

def create_window_data_file(type, directory, year, district, stations, datetime_begin, datetime_end, stride):
  ''' Save the pairwise training/testing historical data and its respective forecast data '''
  print("I am in create window file")
  historical,forecast = create_data(directory=directory,
                                                          number_stations=stations,
                                                          datetime_begin=datetime_begin,
                                                          datetime_end=datetime_end,
                                                          stride=stride)
  data_npz = 'single'+type+'_year{}_district{}_stations{}_days{}_period{}min_history{}s_forecast{}s_stride{}s.npz'.format(
  year,
  district,
  stations,
  (datetime_end-datetime_begin).days,
  SAMPLE_PERIOD.seconds // 60,
  NUM_HISTORICAL_SAMPLES,
  NUM_FORECAST_SAMPLES,
  NUM_STRIDE_SAMPLES
  )

  save_data(data_npz, historical, forecast)



# Create training data
if __name__ == '__main__':
  print("I am in main")
  create_window_data_file(type=TRAINING_TYPE,
            directory=TRAINING_DIRECTORY,
            year=TRAINING_YEAR,
            district=TRAINING_DISTRICT,
            stations=NUM_TRAINING_STATIONS,
            datetime_begin=TRAINING_DATETIME_BEGIN,
            datetime_end=TRAINING_DATETIME_END,
            stride=NUM_STRIDE_SAMPLES)
