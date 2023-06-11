#https://www.kaggle.com/datasets/amirmotefaker/stock-market-analysis-data

import io

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import calendar

data_path = 'Stock_Market_Analysis_Data/'
df = pd.read_csv(data_path + 'stocks.csv')

st.title('Stock Market Analysis')

mnu = st.sidebar.selectbox('메뉴', options=['설명', 'EDA', '시각화', '모델링'])

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

    st.markdown('- (주식 데이터 shape)')
    st.text(f'{df.shape}')

    st.markdown('- Ticker')
    st.text(f"{df['Ticker'].unique()}")

    st.markdown('- 주식 데이터')
    st.dataframe(df.head(10))

    st.markdown('- Data Dtypes')
    st.text(f'{df.dtypes}')

    st.markdown('- Nunique')
    st.dataframe(df.nunique())

    st.markdown('- Df.info')
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # buffer.truncate(0)

elif mnu == '시각화':

    import seaborn as sns
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import plotly.express as px

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # feature_engineering()

    st.subheader('시각화')

    #############################################################
    fig = px.line(df, x='Date',
                  y='Close',
                  color='Ticker',
                  title="3개월간 주식시장 실적")
    st.plotly_chart(fig, use_container_width=True)

    ########################################################
    fig = px.area(df, x='Date', y='Close', color='Ticker',
                  facet_col='Ticker',
                  labels={'Date': 'Date', 'Close': 'Closing Price', 'Ticker': 'Company'},
                  title='Apple, Microsoft, Netflix 및 Google의 주가')
    st.plotly_chart(fig, use_container_width=True)

    ############################################################
    df['MA10'] = df.groupby('Ticker')['Close'].rolling(window=10).mean().reset_index(0, drop=True)
    df['MA20'] = df.groupby('Ticker')['Close'].rolling(window=20).mean().reset_index(0, drop=True)

    for ticker, group in df.groupby('Ticker'):
        fig = px.line(group, x='Date', y=['Close', 'MA10', 'MA20'],
                      title=f"{ticker} 평균 이동량")
        st.plotly_chart(fig, use_container_width=True)

    #############################################
    df['Volatility'] = df.groupby('Ticker')['Close'].pct_change().rolling(window=10).std().reset_index(0, drop=True)
    fig = px.line(df, x='Date', y='Volatility',
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

    x = df.loc[df['Ticker'] == x1, ['Date', 'Close']].rename(columns={'Close': x1})
    y = df.loc[df['Ticker'] == y1, ['Date', 'Close']].rename(columns={'Close': y1})
    df_corr = pd.merge(x, y, on='Date')

    fig = px.scatter(df_corr, x=x1, y=y1,
                     trendline='ols',
                     title=x1+'와(과) '+y1+' 사이의 상관관계')
    st.plotly_chart(fig, use_container_width=True)

    ####################################################################
    st.markdown('상관관계 매트릭스 히트맵')
    correlation_matrix = df.corr()
    plt.figure(figsize=(15, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
    plt.title("Correlation Matrix Heatmap")
    st.pyplot()

    # st.markdown('#### 분석 정리 및 모델링 전략')
    # st.markdown(
    #     '**1. 타깃값 변환:** 분포도 확인 결과 타깃값인 count가 0근처로 치우쳐 있으므로 로그변환하여 정규분포에 가깝게 만들어야한다. 마지막에 다시 지수변환해 count로 복원해야 한다.')
    # st.markdown(
    #     '**2. 파생피처 추가:** datetime 피처는 여러가지 정보의 혼합체이므로 각각을 분리해 year, month, dat, hour, minute, second 피처를 생성할 수 있다.')
    # st.markdown('**3. 파생피처 추가:** datetime 에 숨어 있는 또 다른 정보인 요일(weekday)피처를 추가한다.')
    # st.markdown('**4. 피처 제거:** 테스트 데이터에 없는 피처는 훈련에 사용해도 큰 의미가 없다. 따라서 훈련 데이터에만 있는 casual과 registered 피처는 제거한다.')
    # st.markdown('**5. 피처 제거:** datetime 피처는 인덱스 역할만 하므로 타깃값 예측에 아무런 도움이 되지 않는다.')
    # st.markdown('**6. 피처 제거:** date 피처가 제공하는 정보도 year, month, day 피처에 담겨있다.')
    # st.markdown('**7. 피처 제거:** month는 season 피처의 세부 분류로 볼 수 있다. 데이터가 지나치게 세분화되어 있으면 분류별 데이터수가 적어서 학습에 오히려 방해가 되기도 한다.')
    # st.markdown('**8. 피처 제거:** 막대 그래프 확인 결과 day는 분별력이 없다.')
    # st.markdown('**9. 피처 제거:** 막대 그래프 확인 결과 minute와 second에는 아무런 정보가 담겨 있지 않다.')
    # st.markdown('**10. 이상치 제거:** 포인트 플롯 확인 결과 weather가 4인 데이터는 이상치이다.')
    # st.markdown('**11. 피처 제거:** 산점도 그래프와 히트맵 확인 결과 windspeed 피처에는 결측값이 많고 대여 수량과의 상관관계가 매우 약하다.')

elif mnu == '모델링':

    data_path = 'bike_sharing_demand/'
    train = pd.read_csv(data_path + 'train.csv')
    test = pd.read_csv(data_path + 'test.csv')
    submission = pd.read_csv(data_path + 'sampleSubmission.csv')

    st.markdown('#### 피처 엔지니어링')
    st.markdown('**이상치 제거**')
    st.write('폭우, 번개 속에서 실제로 자전거를 대여했을 수도 있지만 이 한 건의 데이터가 머신러닝 훈련에 부정적 영향을 끼치기 때문에 제거하는 것이 좋습니다.')
    train = train[train['weather'] != 4]
    st.code("train = train[train['weather']!=4]")

    st.markdown('**데이터 합치기**')
    st.write('훈련 데이터와 테스트 데이터에 같은 피처 엔지니어링을 적용하기 위해 두 데이터를 합친다.')
    all_data = pd.concat([train, test], ignore_index=True)
    st.code("all_data = pd.concat([train, test])")

    st.dataframe(all_data)

    st.markdown('**파생 피처 추가**')

    all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])
    all_data['year'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0])
    all_data['month'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1])
    all_data['hour'] = all_data['datetime'].apply(lambda x: x.split()[1].split(':')[0])
    all_data['weekday'] = all_data['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())

    st.code('''
    # 날짜 피처 생성
    all_data['date'] = all_data['datetime'].apply(lambda x: x.split()[0])
    # 연도 피처 생성
    all_data['year'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[0])
    # 월 피처 생성
    all_data['month'] = all_data['datetime'].apply(lambda x: x.split()[0].split('-')[1])
    # 시간 피처 생성
    all_data['hour'] = all_data['datetime'].apply(lambda x: x.split()[1].split(':')[0])
    # 요일 피처 생성
    all_data['weekday'] = all_data['date'].apply(lambda x: datetime.strptime(x, '%y-%m-%d').weekday())
    ''')
    st.write(
        '훈련데이터는 매달 1일부터 19일까지의 기록이고, 테스트 데이터는 매달 20일부터 월말까지의 기록이다. 그러므로 대여 수량을 예측할 때 일(day) 피처는 사용할 필요가 없다. minute와 second 피처도 모든 기록이 같은 값이므로 예측에 사용할 필요가 없다.')

    st.markdown('**필요 없는 피처 제거**')
    drop_features = ['casual', 'registered', 'datetime', 'date', 'windspeed', 'month']
    all_data = all_data.drop(drop_features, axis=1)

    st.code('''
    drop_features = ['casual', 'registered', 'datetime', 'date', 'windspeed', 'month']
    all_data = all_data.drop(drop_features, axis=1)
    ''')

    st.markdown('**피처 선택이란?**')
    st.markdown('모델링 시 데이터의 특징을 잘 나타내는 주요 피처만 선택하는 작업을 **피처 선택( feature selection)** 이라고 한다.')
    st.markdown('피처 선택은 머신러닝 모델 성능에 큰 영향을 준다. 타깃값 예측과 관련없는 피처가 많다면 오히려 성능이 떨어진다.')

    st.markdown('**데이터 나누기**')
    train = all_data[~pd.isnull(all_data['count'])]
    test = all_data[pd.isnull(all_data['count'])]

    X_train = train.drop(['count'], axis=1)
    X_test = test.drop(['count'], axis=1)
    Y_train = train['count']

    st.code('''
    train = all_data[~pd.isnull(all_data['count'])]
    test = all_data[pd.isnull(all_data['count'])]

    X_train = train.drop(['count'], axis=1)
    X_test = test.drop(['count'], axis=1)
    Y_train = train['count']
    ''')

    st.dataframe(X_train.head())

    st.markdown('#### 평가지표 계산 함수 작성')
    st.write('머신러닝 훈련이 제대로 이루어졌는지 확인하려면 대상 능력을 평가할 수단, 즉 평가지표가 필요하다.')
    st.write('요구한 평가지표인 RMSLE를 계산하는 함수부터 만든다.')
    st.markdown('$\displaystyle\sqrt{{1\over{N}}\sum_{i=1}^{N}(\log{(y_{i}+1)}-\ log{(y_{i}+1)})^2}$')
    import numpy as np


    def rmsle(y_true, y_pred, convertExp=True):
        if convertExp:
            y_true = np.exp(y_true)
            y_pred = np.exp((y_pred))

        log_true = np.nan_to_num(np.log(y_true + 1))
        log_pred = np.nan_to_num(np.log(y_pred + 1))

        output = np.sqrt(np.mean((log_true - log_pred) ** 2))
        return output


    st.code('''
    import numpy as np

    def rmsle(y_true, y_pred, convertExp=True):

        if convertExp:
            y_true = np.exp(y_true)
            y_pred = np.exp((y_pred))

        log_true = np.nan_to_num(np.log(y_true + 1))
        log_pred = np.nan_to_num(np.log(y_pred + 1))

        output = np.sqrt(np.mean((log_true - log_pred) ** 2))
        return output
    ''')

    st.markdown('#### 모델 훈련')
    st.write('사이킷런이 제공하는 가장 간단한 선형 회귀 모델인 **LinearRegression 모델** 을 사용한다.')
    st.write('선형회귀에 대한 개념 이해 하려면 아래 링크의 문서를 읽어 보기 바란다.')
    st.markdown('[위키피디아: 선형회귀](https://ko.wikipedia.org/wiki/%EC%84%A0%ED%98%95_%ED%9A%8C%EA%B7%80)')
    st.write('선형회귀는 값을 예측하는 경우 주로 사용된다.')

    from sklearn.linear_model import LinearRegression

    linear_reg_model = LinearRegression()

    log_y = np.log(Y_train)
    linear_reg_model.fit(X_train, log_y)

    st.code('''
    from sklearn.linear_model import LinearRegression

    linear_reg_model = LinearRegression()

    log_y = np.log(Y_train)
    linear_reg_model.fit(X_train, log_y)
    ''')
    st.markdown('')
    st.markdown('확실하게 짚어보고 갑시다.')
    st.markdown('---')
    st.markdown('**훈련 :** 피처(독립변수)와 타깃값(종속변수)이 주어졌을 때 최적의 가중치(회귀계수)를 찾는 과정')
    st.markdown('**예측 :** 최적의 가중치를 아는 상태(훈련된 모델)에서 새로운 독립변수(데이터)가 주어졌을 때 타깃값을 추정하는 과정')

    st.markdown('')
    st.markdown('**탐색적 데이터 분석 :** 예측에 도움이 될 피처를 추리고, 적절한 모델링 방법을 탐색하는 과정')
    st.markdown('**피처 엔지니어링 :**추려진 피처들을 훈련에 적합하도록, 성능 향상에 도움되도록 가공하는 과정')
    st.markdown('---')

    st.markdown('#### 모델 성능 검증')

    preds = linear_reg_model.predict(X_train)
    rmsle_value = rmsle(log_y, preds, True)

    st.code('''
    preds = linear_reg_model.predict(X_train)
    rmsle_value = rmsle(log_y, preds, True)
    ''')

    st.write(f'선형 회귀의 RMSLE 값: {rmsle_value:.4f}')

    st.write('')
    st.markdown('#### 예측 및 결과 제출')

    st.write('1. 테스트 데이터로 예측한 결과를 이용해야 한다.')
    st.write('2. 에측한 값에 지수변환을 해줘야 한다.')

    lineararg_preds = linear_reg_model.predict(X_test)
    submission['count'] = np.exp(lineararg_preds)
    submission.to_csv('submission.csv', index=False)

    st.code('''
    # 테스트 데이터로 예측
    lineararg_preds = linear_reg_model.predict(X_test)
    # 지수 변환
    submission['count'] = np.exp(lineararg_preds)
    # 파일로 저장
    submission.to_csv('submission.csv', index=False)
    ''')

    df_s = pd.read_csv('submission.csv')
    st.dataframe(df_s)