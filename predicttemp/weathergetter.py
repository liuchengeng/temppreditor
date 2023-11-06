import requests

class weatherGetter:
    def __init__(self,location):
        self.location=location
    def apicall(self,url,headers):
        response=requests.get(url,headers=headers)
        data=response.json()
        return data


    def nwspoint(self):
        lat=self.location['lat']
        lon=self.location['long']
        url = f"https://api.weather.gov/points/{lat},{lon}"
        headers = {
            "User-Agent": "(weather_getter, wolfetechhacks@gmail.com)"
        }

        return self.apicall(url, headers)
    def nwsweather(self):
        points=self.nwspoint()
        office=points['properties']['gridId']
        gridx= points['properties']['gridX']
        # print(gridx)
        gridy = points['properties']['gridY']
        # print(gridy)
        # url = f"https://api.weather.gov/gridpoints/OKX/{gridx},{gridy}/stations"
        url = f"https://api.weather.gov/stations/KNYC/observations"
        headers = {
            "User-Agent": "(weather_getter, liuchengeng1@gmail.com)"
        }

        return self.apicall(url, headers)
    def openmeteo(self):
        headers=None
        url=f'https://archive-api.open-meteo.com/v1/archive?latitude=40.7143&longitude=-74.006&start_date=2018-01-01&end_date=2023-09-27&daily=weathercode,temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,windgusts_10m_max,shortwave_radiation_sum&timezone=America%2FNew_York'
        return self.apicall(url,headers)

