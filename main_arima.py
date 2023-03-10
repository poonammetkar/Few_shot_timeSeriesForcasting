# https://www.section.io/engineering-education/multivariate-time-series-using-auto-arima/#prerequisites
# This one uses a different version of arima, but it works the same. The code is slightly different when calling model.predict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

import statsmodels.api as sm # [!] Need to download this library for ARIMA!
from pmdarima import auto_arima # [!] Need to download this library for AUTOARIMA!

import os
import math
import numpy as np
import random
from datetime import datetime, timezone, timedelta
import pytz


def naive_to_aware(datetime):
    ''' Converts the naive datetime.datetime to a aware Pandas TimeStamp for compatibility '''
    return pd.Timestamp(datetime).replace(tzinfo=pytz.UTC)



TRAINING_DISTRICT = 3
TRAINING_YEAR = 2022
STATION_DATA_DIR = r'C:/Users/poonammetkar/Documents/one_shot_time_series_forecast_tinker/preprocessed_data/district_3/year_2023/pickle'#'G:\one_shot_time_series_forecast_tinker\preprocessed_data\district_{}\year_{}\pickle'.format(TRAINING_DISTRICT, TRAINING_YEAR)

DATETIME_BEGIN = naive_to_aware(datetime(year=TRAINING_YEAR, month=3, day=1, hour=0, minute=0))
DATETIME_END = naive_to_aware(datetime(year=TRAINING_YEAR, month=6, day=1, hour=0, minute=0))
SAMPLE_PERIOD = timedelta(minutes=30)
SEASONALITY = timedelta(hours=24) # Let's choose one day as our seasonal cycle (ideally it would be 1 week, but ARIMA fails for a large sample of seasonal data)

HISTORICAL_WINDOW = timedelta(days=3, hours=0, minutes=0)
FORECAST_WINDOW = timedelta(days=0, hours=12, minutes=0)
STRIDE_WINDOW = timedelta(days=0, hours=12, minutes=0)



def adf_test(df):
    # Augmented dickey-fuller test to see the components of our data
    adf_test = adfuller(df, autolag='AIC')
    print("1. ADF: ", adf_test[0])
    print("2. P-Value: ", adf_test[1])
    print("3. Num of Lags: ", adf_test[2])
    print("4. Num of Observations Used for ADF Regression: ", adf_test[3])
    print("5. Critical Values: ")
    for key, val in adf_test[4].items():
            print("\t", key, ": ", val)

    # The data we are dealing with is an additive model because the seasonal component is roughly independent of the trend
    WEEK_PERIOD = int((DATETIME_END - DATETIME_BEGIN) / (timedelta(days=7))) # Ideally, the main seasonality is week
    decomp = seasonal_decompose(df, model='additive', period=WEEK_PERIOD)
    decomp.plot()
    plt.show()



def load_station_data(filename=None):
    if filename:
        return np.load(os.path.join(STATION_DATA_DIR, filename), allow_pickle=True)
    else:
        # Just choose the first file if no filename is specified
        return np.load(os.path.join(STATION_DATA_DIR, os.listdir(STATION_DATA_DIR)[0]), allow_pickle=True)



if __name__ == '__main__':
    # Load dataframe and just grab a subset of time from it
    df = load_station_data()
    df = df.resample(SAMPLE_PERIOD).sum()
    subset_df = df[DATETIME_BEGIN:DATETIME_END]

    # Run Augmented Dickey-Fuller Test to see the compositional components of our data
    adf_test(subset_df)

    # Here, we run auto_arima to find the order parameters that give the most accurate ARIMA model
    # We do not not need to run this multiple times for the same training data, as it would take a long time to run
    # I ran it once for this instance's training data and got an order of (5, 0, 1) as the best parameters based on a low  Akaike Information Critera (AIC) score.
    # I will use that when creating the model in the lines below
    '''
    stepwise_fit = auto_arima(subset_df['total_flow'], m=int(SEASONALITY/SAMPLE_PERIOD), start_p=0, start_q=0, test='adf', seasonal=True, max_order=4, trace=True, suppress_warnings=True)
    '''

    # Starting from the beginning date and getting N-windows each separated by a stride from each other,
    # Train on the historical data (historical_df) and forecast
    # Compare the predicted and actual forecast to each other
    # Accumulate the absolute errors and then divide by N to get the mean absolute error (the performance metric of ARIMA)
    # Note that this example just takes a few windows of one station
    # You can modify the code to take a random number of windows of multiple stations
    # For example, MAE from 50 random windows of 1000 random stations
    num_windows_checked = 0 # Can precalculate this value in case of error_accumulate overflow (more on that explained later in the lines below)
    N = 5
    error_accumulate = np.zeros(int(FORECAST_WINDOW/SAMPLE_PERIOD))
    for i in range(0, N):
        # Note that ARIMA models cannot be trained on one station and generalize to other stations
        # Therefore, we cannot train on a year's worth of station data. We are assuming we only have a few days of data available (scarce data).
        # If we want an accurate comparison, we should train on 7 days of data only if we want to compare it to our Siamese Model (which only uses 7 days of data at least)
        # Yes, even though the historical horizon of our Siamese horizon may be less than a week, say 3 days, we run a sliding window with a stride through the week of data to find the best match!
        # However, this ARIMA model cannot handle multivariate data (It does not know what a weekday vs. weekend is because it has no weekly feature)
        # To give it a fairer shot, we will train it on 14 days of data so both the daily and weekly seasonal components are captured
        # The order of this model is found by calling auto_arima in the line above (currently commented out)
        TRAIN_WINDOW = timedelta(days=14)
        historical_df = subset_df[DATETIME_BEGIN:DATETIME_BEGIN+TRAIN_WINDOW+STRIDE_WINDOW*i]
        actual_forecast_df = subset_df[DATETIME_BEGIN+TRAIN_WINDOW+STRIDE_WINDOW*i:DATETIME_BEGIN+TRAIN_WINDOW+STRIDE_WINDOW*i+FORECAST_WINDOW]

        # Error Check for any windows with NAN or INF values and skip them
        all_valid_data = historical_df.isna().sum().sum() == 0 and np.isinf(historical_df).sum().sum() == 0 and \
                            actual_forecast_df.isna().sum().sum() == 0 and np.isinf(actual_forecast_df).sum().sum() == 0
        if not all_valid_data:
            print("There exists NAN or INF in the data")
            print("\nMissing values :  ", historical_df.isnull().any())
            print("\nMissing values :  ", actual_forecast_df.isnull().any())
            continue

        # Create the model
        # [!] one major improvement in accuracy is adding auxiliary "exog" equations to the ARIMA model
        model = sm.tsa.arima.ARIMA(historical_df, order=(5,0,1))
        model = model.fit()

        # After fitting, run the prediction. The forecast will be on the same time range as the actual_forecast_df
        predicted_forecast = model.predict(start=DATETIME_BEGIN+TRAIN_WINDOW+STRIDE_WINDOW*i,
                                           end=DATETIME_BEGIN+TRAIN_WINDOW+STRIDE_WINDOW*i+FORECAST_WINDOW,
                                           return_conf_int=True)
        predicted_forecast_df = pd.DataFrame(predicted_forecast)

        '''
        pd.concat([actual_forecast_df, predicted_forecast_df], axis=1).plot() # just plot the forecast window
        '''
        pd.concat([historical_df, actual_forecast_df, predicted_forecast_df], axis=1).plot()
        plt.show()

        # Take the absolute error of the predictions
        ae_prediction = np.abs(actual_forecast_df['total_flow'] - predicted_forecast_df['predicted_mean'])
        '''
        ae_prediction.plot()
        plt.show()
        '''
        ae_prediction = ae_prediction.to_numpy()

        if not error_accumulate.all():
            error_accumulate = ae_prediction
        else:
            error_accumulate += ae_prediction
            # If there is an overflow, you can scale by the final_num_windows_checked
            # Note that final_num_windows_checked != N due to windows existing with NAN or INF
            # So you should run a for loop before this to count the num_windows_checked via incrementing
            # Then once you get that final_num_windows_checked
            # # error_accumulate += (ae_prediction/final_num_windows_checked)

        num_windows_checked += 1
        print("Index: {}".format(i))
        print("ERROR_ACCUMULATE: {}".format(error_accumulate))

    # Calculate and plot the MAE
    mae = error_accumulate/num_windows_checked
    print("MAE: ")
    print(mae)

    mae_df = pd.DataFrame(mae, columns=['MAE'])
    mae_df.plot()
    plt.title("MAE: {} sequential windows of 1 station".format(N))
    plt.xlabel('Forecast Horizon')
    plt.ylabel('Vehicles')
    plt.show()
