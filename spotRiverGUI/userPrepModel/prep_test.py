from river import compose
from river import preprocessing
from river import feature_extraction
from river import stats
import numbers

print("Prep model for bike sharing demand prediction")
# season,year,month,hour,holiday,weekday,workingday,weather,temp,feel_temp,humidity,windspeed,count
# x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,y
# x1 = season
# x2 = year
# x3 = month
# x4 = hour
# x5 = holiday
# x6 = weekday
# x7 = workingday
# x8 = weather
# x9 = temp
# x10 = feel_temp
# x11 = humidity
# x12 = windspeed
# y = count


# def set_prep_model():
#     # prepmodel = compose.Select('humidity', 'temp', 'feel_temp', 'windspeed')
#     num = compose.Select("x11", "x9", "x10", "x12")
#     num += (
#          feature_extraction.TargetAgg(by=["x3"], how=stats.Mean())
#     )
#     num = num | preprocessing.StandardScaler()
#     cat = compose.SelectType(str) | preprocessing.OneHotEncoder()
#     prepmodel = (num + cat)
#     return prepmodel


def set_prep_model():
    num = compose.Select("x11", "x9", "x10", "x12") | preprocessing.StandardScaler()
    cat = compose.SelectType(str) | preprocessing.OneHotEncoder()
    prepmodel = num + cat
    return prepmodel


# def set_prep_model():
#     prepmodel = preprocessing.MinMaxScaler()
#     return prepmodel
