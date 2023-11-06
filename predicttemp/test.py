import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
data=pd.read_csv('C:/Users/86188/Desktop/ww.csv')
data=data.iloc[:,2:6].join(data.iloc[:,7:9])
data['DATE']=pd.to_datetime(data['DATE'])
data['year']=data['DATE'].dt.year.apply(lambda x: int(x))
data['month']=data['DATE'].dt.month.apply(lambda x: int(x))
data['day']=data['DATE'].dt.day.apply(lambda x: int(x))
data['one']=data['TMAX'].shift(1)
data['two']=data['TMAX'].shift(2)
data['TMIN']=data['TMIN'].shift(1)
data['AWND']=data['AWND'].shift(1)
data['PRCP']=data['PRCP'].shift(1)
data['SNOW']=data['SNOW'].shift(1)
data_clean=data.dropna()
#
label=np.array(data_clean['TMAX'])
data_clean=data_clean.drop('TMAX',axis=1)
data_clean=data_clean.drop('DATE',axis=1)
feature_list = list(data_clean.columns)

data_clean= np.array(data_clean)

train_x,test_x,train_y,test_y=train_test_split(data_clean,label,test_size=0.25,random_state=42)
# param_grid = {
#     'n_estimators': [1000, 2000, 3000],
#     'max_depth': [None, 10, 20],
#     'min_samples_leaf': [1, 2, 4]
# }
# rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
#                            param_grid=param_grid,
#                            cv=5)
rf = RandomForestRegressor(n_estimators=1000, random_state=42,min_samples_leaf=6)

# 训练
rf.fit(train_x, train_y)
prediction=rf.predict(test_x)
mse=mean_squared_error(test_y,prediction)
importance=list(rf.feature_importances_)
feature_importance=[(feature,importance ) for feature ,importance in zip(feature_list,importance)]
print(mse)

recent_true=np.array(data[['TMAX']].tail(30)).ravel()
recent_pre=rf.predict(data_clean)[-30:]
recent_date=np.array(data[['DATE']].tail(30)).ravel()


plt.scatter(recent_date,recent_true,label='actual',marker='o',color='blue')
plt.scatter(recent_date,recent_pre,label='pre',marker='x',color='red')
plt.title('actual vs pre')
plt.xlabel('date')
plt.ylabel('temp')

# 添加图例
plt.legend()

# 旋转x轴标签以更好地显示日期
plt.xticks(rotation=45)

# 显示图形
plt.tight_layout()
plt.show()

