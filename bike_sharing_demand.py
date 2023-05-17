import io

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import calendar

data_path = 'bike_sharing_demand/'
train = pd.read_csv(data_path + 'train.csv',parse_dates=['datetime'])
test = pd.read_csv(data_path + 'test.csv', parse_dates=['datetime'])
submission = pd.read_csv(data_path + 'sampleSubmission.csv', parse_dates=['datetime'])
