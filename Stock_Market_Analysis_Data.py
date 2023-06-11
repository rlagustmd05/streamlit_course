#https://www.kaggle.com/datasets/amirmotefaker/stock-market-analysis-data

import io

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import calendar
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.express as px
# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

data_path = 'Stock_Market_Analysis_Data/'
train = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'test.csv')



st.title('Stock Market Analysis')

mnu = st.sidebar.selectbox('메뉴', options=['설명', 'EDA', '시각화', '예측'])

if mnu == '설명':
    st.subheader('데이터 분석 목표')
    st.write('''
    3개월 동안 Apple, Microsoft, Netflix 및 Google의 과거 주가 데이터가 
    주어졌을 때의 목표는 다양한 데이터 과학 기술을 사용하여 주식 시장에서 
    위 회사들의 성과를 분석하고 비교하는 것입니다.
    
    구체적으로 주가 움직임의 추세와 패턴을 파악하고 기업별 이동 평균과 변동성을 
    계산하며 상관관계 분석을 통해 서로 다른 주가 간의 관계를 살펴보는 것이 목표입니다.
    ''')
    st.image('Stock.jpg')

    st.markdown('#### 데이터 필드')
    st.markdown('**Ticker** - 종목 코드')
    st.markdown('**Date** - 날짜')
    st.markdown('**Open** - 시가')
    st.markdown('**High** - 고가')
    st.markdown('**Low** - 저가')
    st.markdown('**Close** - 종가')
    st.markdown('**Adj Close** - 수정 종가')
    st.markdown('**Volume** - 거래량')

elif mnu == 'EDA':
    st.subheader('EDA')

    st.markdown('- (훈련 데이터 shape, 테스트 데이터 shape)')
    st.text(f'({train.shape}), ({test.shape})')

    st.markdown('- Ticker')
    st.text(f"{train['Ticker'].unique()}")

    st.markdown('- 훈련 주식 데이터')
    st.dataframe(train.head(10))
    st.markdown('- 테스트 주식 데이터')
    st.dataframe(test.head(10))

    st.markdown('- Data Dtypes')
    st.text(f'{train.dtypes}')

    st.markdown('- 훈련 & 테스트 Nunique')
    st.dataframe(train.nunique())

    st.markdown('- train.info')
    buffer = io.StringIO()
    train.info(buf=buffer)
    st.text(buffer.getvalue())
    st.markdown('- test.info')
    buffer = io.StringIO()
    test.info(buf=buffer)
    st.text(buffer.getvalue())

elif mnu == '시각화':

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.subheader('시각화')

    #############################################################
    fig = px.line(train, x='Date',
                  y='Close',
                  color='Ticker',
                  title="3개월간 주식시장 실적")
    st.plotly_chart(fig, use_container_width=True)

    ########################################################
    fig = px.area(train, x='Date', y='Close', color='Ticker',
                  facet_col='Ticker',
                  labels={'Date': 'Date', 'Close': 'Closing Price', 'Ticker': 'Company'},
                  title='Apple, Microsoft, Netflix 및 Google의 주가')
    st.plotly_chart(fig, use_container_width=True)

    ############################################################
    train['MA10'] = train.groupby('Ticker')['Close'].rolling(window=10).mean().reset_index(0, drop=True)
    train['MA20'] = train.groupby('Ticker')['Close'].rolling(window=20).mean().reset_index(0, drop=True)

    for ticker, group in train.groupby('Ticker'):
        fig = px.line(group, x='Date', y=['Close', 'MA10', 'MA20'],
                      title=f"{ticker} 평균 이동량")
        st.plotly_chart(fig, use_container_width=True)

    #############################################
    train['Volatility'] = train.groupby('Ticker')['Close'].pct_change().rolling(window=10).std().reset_index(0, drop=True)
    fig = px.line(train, x='Date', y='Volatility',
                  color='Ticker',
                  title='Apple, Microsoft, Netflix 및 Google의 변동성')
    st.plotly_chart(fig, use_container_width=True)

    ###################################################
    X_line = st.radio(
        "x축 종목 코드",
        ('AAPL', 'MSFT', 'NFLX', 'GOOG'),horizontal = True)

    if X_line == 'AAPL':
        x1 = 'AAPL'
        Y_line = st.radio(
            "y축 종목 코드",
            ('MSFT', 'NFLX', 'GOOG'), horizontal=True)
    elif X_line == 'MSFT':
        x1 = 'MSFT'
        Y_line = st.radio(
            "y축 종목 코드",
            ('AAPL', 'NFLX', 'GOOG'), horizontal=True)
    elif X_line == 'NFLX':
        x1 = 'NFLX'
        Y_line = st.radio(
            "y축 종목 코드",
            ('AAPL', 'MSFT', 'GOOG'), horizontal=True)
    else:
        x1 = 'GOOG'
        Y_line = st.radio(
            "y축 종목 코드",
            ('AAPL', 'MSFT', 'NFLX'), horizontal=True)

    if Y_line == 'AAPL':
        y1 = 'AAPL'
    elif Y_line == 'MSFT':
        y1 = 'MSFT'
    elif Y_line == 'NFLX':
        y1 = 'NFLX'
    else:
        y1 = 'GOOG'

    x = train.loc[train['Ticker'] == x1, ['Date', 'Close']].rename(columns={'Close': x1})
    y = train.loc[train['Ticker'] == y1, ['Date', 'Close']].rename(columns={'Close': y1})
    df_corr = pd.merge(x, y, on='Date')

    fig = px.scatter(df_corr, x=x1, y=y1,
                     trendline='ols',
                     title=x1+'와(과) '+y1+' 사이의 상관관계')
    st.plotly_chart(fig, use_container_width=True)

    ####################################################################
    st.markdown('상관관계 매트릭스 히트맵')
    correlation_matrix = train.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    st.pyplot()

elif mnu == '예측':

    st.subheader('예측')

    from sklearn.preprocessing import MinMaxScaler

    prev_close_value = test[test['Date'] > '2010-5-10']

    scaler = MinMaxScaler(feature_range=(0, 1))
    closed_values = np.array(prev_close_value['Close']).reshape(len(prev_close_value), 1)
    norm_closed = scaler.fit_transform(closed_values)

    sns.histplot(norm_closed, kde=True)
    st.pyplot()
    st.markdown('#### Norm_closed shape')
    st.text(f'{norm_closed.shape}')

    st.markdown('#### Prev_close_value shape')
    st.text(f'{prev_close_value.shape}')

    time_stamp = 15

    def time_stamps(norm_closed):
        X_data = []
        y_data = []
        for i in range(len(norm_closed) - time_stamp - 1):
            X_data.append(norm_closed[i:i + time_stamp, 0].flatten())
            y_data.append(norm_closed[i + time_stamp])
        return np.array(X_data), np.array(y_data)


    def train_test(x, split=0.8):
        train_size = (int)(len(x) * (split))
        test_size = len(x) - train_size
        train_data = x[:train_size, :]
        test_data = x[train_size:, :]
        return np.array(train_data), np.array(test_data)


    train_data, test_data = train_test(norm_closed)

    x_train, y_train = time_stamps(train_data)
    x_test, y_test = time_stamps(test_data)

    st.markdown('#### Train_X & Y')
    st.markdown(x_train.shape)
    st.markdown(y_train.shape)
    st.markdown('#### Test_X & Y')
    st.markdown(x_test.shape)
    st.markdown(y_test.shape)

    st.markdown('#### x_test')
    x_test
    st.markdown('#### y_test')
    y_test

    from xgboost import XGBRegressor

    my_model = XGBRegressor(n_estimators=1000)
    my_model.fit(x_train, y_train, verbose=False)

    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import math

    st.markdown('#### Predictions on test data')
    test_pred = my_model.predict(x_test)
    st.markdown(f'RMSE {math.sqrt(mean_squared_error(y_test, test_pred))}')

    st.markdown('#### Predictions on train data')
    train_pred = my_model.predict(x_train)
    st.markdown(f'RMSE {math.sqrt(mean_squared_error(y_train, train_pred))}')

    st.markdown('#### Prev_close_value - Train_pred - Test_pred')
    len(prev_close_value) - len(train_pred) - len(test_pred)

    st.markdown('#### Train DataFrame')
    original_train_y_value = scaler.inverse_transform(np.array(train_pred).reshape(-1, 1))
    pred_train_value = np.array([np.nan] * (train_data))
    pred_train_value[time_stamp:-1] = original_train_y_value[:]
    pred_train_value.size

    res_df = pd.DataFrame()
    res_df['Date'] = prev_close_value['Date'].iloc[:len(train_data)]
    res_df['Train_closed'] = pred_train_value
    res_df

    st.markdown('#### Test DataFrame')
    original_test_y_value = scaler.inverse_transform(np.array(test_pred).reshape(-1, 1))
    pred_test_value = np.array([np.nan] * (test_data))
    pred_test_value[time_stamp:-1] = original_test_y_value[:]

    res_df_test = pd.DataFrame()
    res_df_test['Date'] = prev_close_value['Date'].iloc[len(train_data):]
    res_df_test['Test_closed'] = pred_test_value
    res_df_test

    st.markdown('#### Original_test_y_value')
    original_test_y_value

    st.markdown('#### Prev_close_value')
    prev_close_value

    st.markdown("#### <Axes: xlabel='Date', ylabel='Close'>")
    plt.figure(figsize=(16, 8))
    sns.lineplot(x=prev_close_value['Date'], y=prev_close_value['Close'], label='Original data')
    sns.lineplot(x=res_df['Date'], y=res_df['Train_closed'], label='Predicted train data')
    sns.lineplot(x=res_df_test['Date'], y=res_df_test['Test_closed'], label='Predicted test data')
    st.pyplot()