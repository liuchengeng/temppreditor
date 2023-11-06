Daily Max Temperature Predictor for New York City
 --------------------------------------------------------------
In this project , I built 5 different models to predict daily max temperature in New York(Central Park) using data from 5 different sources.
These models are based on regression algorithms,get today's specific weather data ,each of the model will tell his prediction of tomorrow's max temperature of New York.
-----------------------------------------------------------------
datagetter.py:
Incuding a api caller, a csv file reader and 5 different free and useful weather data api that are ready to use. 
Sources includes:
OpenMeteo Historical Weather Data
Ensemble Model Predictions(collection of weather prediction of different models) 
Weather API
MeteoStat Historical Data
Hourly Weather Data(MeteoStat)
Data of these sources are used to the models.
------------------------------------------------------------
kalshiapi.py:
It can be used to check the information of different series,event,markets,check the infromation of the loged-in account. It can also be used to create orders
-------------------------------------------------------------
predictor.py:
The TemperaturePredictorclass provides functionalities to predict temperature based on the provided dataset using the XGBoost algorithm.
Functions includes training the model, evaluate the model, visualize the output plot scatter graphs of prediction vs. groud truth ,prediction based on provided data.
-----------------------------------------------------------
source_ensemble.py:
In thsi model, I gathered prediction data from other predictors as the input of the model.
I only use data from the previous month since that's the limit for a free account. 
----------------------------------------------------------
source_hourly.py:
The data source I used here is the hourly temperature from 0:00 am to 12:00 am. As the max temperature is usually hit at 2 pm , I only use 12 hourly temperature to predict the max temperature. 
----------------------------------------------------------
source_noaa.py:
Weather data downloaded from noaa(National Oceanic and Atmospheric Administration)
----------------------------------------------------------
source_open_his.py:
Weather data from openmeteo api.
---------------------------------------------------------
source_meteostas.py:


NOTICE: I put the main report in source_meteostas.ipynb



Prerequisites: Python 3 

Contact:
chengeng liu
lcg518@bu.edu

