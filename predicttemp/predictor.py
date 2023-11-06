import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from datetime import datetime

class TemperaturePredictor:
    def __init__(self, filepath):
        self.data = self.load_and_preprocess_data(filepath)
        self.model = None

    def load_and_preprocess_data (self,filepath):
        data = pd.read_csv(filepath)
        data = data.iloc[:, 2:6].join(data.iloc[:, 7:9])
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['year'] = data['DATE'].dt.year.apply(lambda x: int(x))
        data['month'] = data['DATE'].dt.month.apply(lambda x: int(x))
        data['day'] = data['DATE'].dt.day.apply(lambda x: int(x))
        data['one'] = data['TMAX'].shift(1)
        data['two'] = data['TMAX'].shift(2)
        data['TMIN'] = data['TMIN'].shift(1)
        data['PRCP'] = data['PRCP'].shift(1)
        data['SNOW'] = data['SNOW'].shift(1)

        data = data.drop('AWND', axis=1)
        annual_avg_temperature = data.groupby(['month', 'day'])['TMAX'].mean().reset_index()
        month_avg = data.groupby(['year', 'month'])['TMAX'].mean().reset_index()
        data = data.merge(annual_avg_temperature, on=['month', 'day'], how='left')
        data = data.merge(month_avg, on=['year', 'month'], how='left')
        data = data.rename(columns={'TMAX_y': 'pastyearave', 'TMAX_x': 'TMAX', 'TMAX': 'monthave'})
        data['3days_ave'] = data['TMAX'].rolling(window=3,min_periods=1).mean().shift(1)
        data['7days_ave'] = data['TMAX'].rolling(window=7, min_periods=1).mean().shift(1)

        return data



    def prepare_training_data(self):

        data_clean = self.data.dropna().drop('DATE', axis=1)

        label = np.array(data_clean['TMAX'])
        data_clean = data_clean.drop('TMAX', axis=1)
        data_clean=data_clean.drop('SNOW',axis=1)

        return data_clean, label

    def train_model(self):
        data_clean, label = self.prepare_training_data()
        train_x, test_x, train_y, test_y = train_test_split(data_clean, label, test_size=0.25, random_state=67)
        self.model = xgb.XGBRegressor(n_estimators=1000, random_state=81, learning_rate=0.01,max_depth=20)
        self.model.fit(train_x, train_y)

    def evaluate_model(self):
        if self.model is None:
            raise Exception("Model is not trained. Call train_model() first.")

        data_clean, label = self.prepare_training_data()
        feature_list =data_clean.columns
        train_x, test_x, train_y, test_y = train_test_split(data_clean, label, test_size=0.25, random_state=67)
        prediction = self.model.predict(test_x)
        mse = mean_squared_error(test_y, prediction)
        mae = mean_absolute_error(test_y, prediction)
        importance = list(self.model.feature_importances_)
        feature_importance = [(feature, importance) for feature, importance in zip(feature_list, importance)]
        print(f'MSE: {mse}',feature_importance)
        print(f'MAE: {mae}')

    def plot(self):
        data=self.data
        data_clean, _ = self.prepare_training_data()
        recent_true = np.array(data[['TMAX']].tail(1000)).ravel()


        recent_pre = self.model.predict(data_clean)[-1000:]
        recent_date = np.array(data[['DATE']].tail(1000)).ravel()
        plt.scatter(recent_date, recent_true, label='actual', color='blue',marker='o')
        plt.scatter(recent_date, recent_pre, label='pre', color='red',marker='x')
        plt.title('actual vs pre')
        plt.xlabel('date')
        plt.ylabel('temp')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def predict_today(self, today):
        today_pre = self.model.predict(today)
        print(f"Prediction for today: {today_pre[0]}")
    def getave(self,date):
        data=self.data
        date_obj = datetime.strptime(date, "%Y-%m-%d")
        month = date_obj.month
        day = date_obj.day
        year=date_obj.year
        pastave = data.loc[(data['month'] == month) & (data['day'] == day), 'pastyearave'].iloc[0]
        print(data[(data['month'] == month) & (data['day'] == day)])
        monthave= data.loc[(data['month'] == month) & (data['year'] == year), 'monthave'].iloc[0]
        print(f'pastave={pastave}\nmonthave={monthave}')





