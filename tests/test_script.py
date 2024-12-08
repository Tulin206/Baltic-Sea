import sys
sys.path.append('C:/Users/Tim/Desktop/ISRAT/RostockUniversity/PyCharmProjects')
import pytest
import numpy as np
from lstm_cnn_with_dvc import load_raw_data

print("test script initiated")
def test_load_raw_data():
    print("test script initiated")
    train, test, mask = load_raw_data("C:/Users/Tim/Desktop/ISRAT/RostockUniversity/PyCharmProjects/data.npz")
    assert train is not None
    assert test is not None
    assert mask is not None
    assert isinstance(train, np.ndarray)
    assert isinstance(test, np.ndarray)
    assert isinstance(mask, np.ndarray)
    print("test script is being executed")
