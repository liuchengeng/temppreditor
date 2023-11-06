from predictor import TemperaturePredictor
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":
    filepath = 'C:/Users/86188/Desktop/ww.csv'
    tp1=TemperaturePredictor(filepath)
    tp1.train_model()


    tp1.evaluate_model()
    tp1.plot()

    # tp1.getave('2023-09-26')

    # today=pd.DataFrame({'PRCP':[0.59],'SNOW':[0],'TMIN':[57],'year':[2023],'month':[9],'day':[26],'one':[66],'two':[66],'pastyearave':[75.5],'monthave':[81.66],'7days_ave':[69]})
    # predict=tp1.predict_today(today)
    # print(predict)