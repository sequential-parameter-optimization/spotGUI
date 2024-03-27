from river import compose
from river import preprocessing
import numbers


def set_prep_model():
    num = compose.SelectType(numbers.Number) | preprocessing.StandardScaler()
    cat = compose.SelectType(str) | preprocessing.OneHotEncoder()
    prepmodel = num + cat
    return prepmodel
