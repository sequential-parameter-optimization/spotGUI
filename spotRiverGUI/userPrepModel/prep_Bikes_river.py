from river import compose
from river import preprocessing
from river import feature_extraction
from river import stats

print("Prep model for the river Bikes dataset")
# See: https://riverml.xyz/latest/examples/bike-sharing-forecasting/
# and
# https://github.com/online-ml/river/blob/main/river/datasets/bikes.py
# Bikes dataset will be stored as toulouse_bikes.csv file with the following columns:
# moment,station,clouds,description,humidity,pressure,temperature,wind, bikes
# x1,x2,x3,x4,x5,x6,x7,x8,y
# x1: moment
# x2: station
# x3: clouds
# x4: description
# x5: humidity
# x6: pressure
# x7: temperature
# x8: wind
# y: bikes


def get_hour(x):
    x["hour"] = x["x1"].hour
    return x


def set_prep_model():
    # prepmodel = compose.Select('clouds', 'humidity', 'pressure', 'temperature', 'wind')
    prepmodel = compose.Select("x3", "x5", "x6", "x7", "x8")
    prepmodel += get_hour | feature_extraction.TargetAgg(by=["x2", "hour"], how=stats.Mean())
    prepmodel |= preprocessing.StandardScaler()
    return prepmodel
