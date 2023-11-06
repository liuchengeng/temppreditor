from datacollection import data_openmeteo
import pandas as pd
import numpy as np

df=pd.DataFrame(data_openmeteo['daily'])
pd.set_option('display.max_columns', None)
print(df.head(10))

lag_features = ['weathercode', 'temperature_2m_min',
                'precipitation_sum', 'windspeed_10m_max', 'windgusts_10m_max',
                'shortwave_radiation_sum']
for feature in lag_features:
    df[f"{feature}_lag1"] = df[feature].shift(1)
df['rolling_std_7d'] = df['temperature_2m_max'].rolling(window=7).std()
df['rolling_max_7d'] = df['temperature_2m_max'].rolling(window=7).max()
df['rolling_min_7d'] = df['temperature_2m_max'].rolling(window=7).min()
df['temp_diff'] = df['temperature_2m_max'].diff()
df['is_thunderstorm'] = df['weathercode'].apply(lambda x: 1 if x == 5 else 0)
df['is_fog'] = df['weathercode'].apply(lambda x: 1 if x == 45 else 0)
df['day_of_week'] = pd.to_datetime(df['time']).dt.dayofweek
df['week_of_year'] = pd.to_datetime(df['time']).dt.isocalendar().week
df['wind_temp_interaction'] = df['windspeed_10m_max'] * df['temperature_2m_max']
df['radiation_temp_interaction'] = df['shortwave_radiation_sum'] * df['temperature_2m_max']
df['temperature_2m_max'] = df['temperature_2m_max'].shift(-1)
data_clean=df.dropna()

print(data_clean)