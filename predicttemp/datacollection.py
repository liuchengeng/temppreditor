import requests
import pandas as pd
from weathergetter import weatherGetter
location = {
     "lat": "40.78333",
     "long": "-73.96667",
}
test = weatherGetter(location)
data_nws= test.nwsweather()
data_openmeteo=test.openmeteo()



