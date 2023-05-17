import io

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import calendar

data_path = 'bike_sharing_demand/'
train = pd.read_csv(data_path + 'train.csv', parse_dates=['datetime'])
test = pd.read_csv(data_path + 'test.csv', parse_dates=['datetime'])
submission = pd.read_csv(data_path + 'sampleSubmission.csv', parse_dates=['datetime'])


def feature_engineering():
    train['year'] = train['datetime'].dt.year
    train['month'] = train['datetime'].dt.month
    train['day'] = train['datetime'].dt.day
    train['hour'] = train['datetime'].dt.hour
    train['minute'] = train['datetime'].dt.minute
    train['second'] = train['datetime'].dt.second
    train['date'] = train['datetime'].dt.date
    train['dayofweek'] = train ['datetime'].dt.dayofweek
    train['season'] = train['season'].map({1: 'spring',
                                           2: 'summer',
                                           3: 'fall',
                                           4: 'winter'})

    train['weather'] = train['weather'].map({1: 'clear',
                                             2: 'mist, few clouds',
                                             3: 'light snow, rain, thunderstorm',
                                             4: 'heavy rain, thunderstorm, snow, fog'})
    train['weekday'] = train['date'].apply(lambda date: calendar.day_name[datetime.combine(date, datetime.min.time()).weekday()])

st.title('Bike Sharing Demand')

mnu = st.sidebar.selectbox('메뉴', options=['설명', 'EDA', '시각화', '모델링'])

if mnu == '설명':
    st.sidebar('요구사항')
    st.write('''
    자전거 공유 시스템은 회원 가입, 대여 및 자전거 반납 프로세스가 도시 전역의 키오스크 위치 네트워크를 통해 자동화되는 자전거 대여 수단입니다. 이러한 시스템을 사용하여 사람들은 한 위치에서 자전거를 빌리고 필요에 따라 다른 위치에 반납할 수 있습니다. 현재 전 세계적으로 500개가 넘는 자전거 공유 프로그램이 있습니다. 이러한 시스템에서 생성된 데이터는 여행 기간, 출발 위치, 도착 위치 및 경과 시간이 명시적으로 기록되기 때문에 연구자에게 매력적입니다. 따라서 자전거 공유 시스템은 도시의 이동성을 연구하는 데 사용할 수 있는 센서 네트워크로 기능합니다. 이 대회에서 참가자는 워싱턴 DC의 Capital Bikeshare 프로그램에서 자전거 대여 수요를 예측하기 위해 과거 사용 패턴과 날씨 데이터를 결합해야 합니다.
    ''')
    st.write('''2년 동안의 시간당 임대 데이터가 제공됩니다. 이 대회의 경우 훈련 세트는 매월 첫 19일로 구성되고 테스트 세트는 20일부터 월말까지입니다. 대여 기간 전에 사용할 수 있는 정보만 사용하여 테스트 세트에서 다루는 각 시간 동안 대여한 총 자전거 수를 예측해야 합니다.''')
    st.image('')