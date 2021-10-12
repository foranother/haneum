import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate('react-with-firebase-f9b6e-firebase-adminsdk-nqjsb-0b44051b21.json')
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://react-with-firebase-f9b6e-default-rtdb.firebaseio.com/'
})

def korea_top_etf(risk) :
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import os
    !pip install finance-datareader
    !pip install tensorflow
    import FinanceDataReader as fdr
    
    korea_etf=pd.read_csv('국내_ETF.csv')
    korea_etf['종목코드'] = korea_etf['종목코드'].astype(str)
    korea_etf.loc[8, '종목코드'] ='091180'
    korea_etf.loc[9, '종목코드'] ='091160'
    korea_etf_risk = korea_etf[korea_etf['투자위험등급']==risk]
    
    
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    

    
    rate_list =[]
    sentiment_list = []

    for code, theme in zip(korea_etf_risk['종목코드'],korea_etf_risk['테마']):
        #감성지수 데이터 불러오기
        point =pd.read_csv(str(theme)+'.csv')
        nlp_point = point[['Date', 'point']]
        nlp_point['Date'] = pd.to_datetime(nlp_point['Date'].str.replace('.', '-'))
    
        #하루에 한개 이상 존재하는 기사->점수를 하루 단위로 평균 
        stock_point=nlp_point.groupby('Date').mean()
        stock_point.reset_index(inplace = True)
    
        #주가데이터 불러오기 위한 날짜 저장 
        start_date=nlp_point.iloc[-1, 0]
        end_date=nlp_point.iloc[0, 0]
    
        #ETF주가 데이터 불러오기
        stock= fdr.DataReader(str(code), start_date, end_date)
        stock.reset_index(inplace = True)
    
        #감성지수와 ETF주가 데이터프레임 결합 
        stock=pd.merge(stock, stock_point, on ='Date', how ='left' )
        print(stock.head())
    
        ##결측치 처리
        #결측치 앞방향으로 채우기
        stock['point'] = stock['point'].fillna(method='ffill')
        #맨 처음 날짜가 결측치인 경우는 0으로 채우기 
        stock['point'] = stock['point'].fillna(0)
    
        ##이상치 처리
        Q1 = stock['point'].quantile(.25)
        Q3 = stock['point'].quantile(.75)
        IQR = Q3 - Q1
        Min = Q1 - 1.5 * IQR
        Max = Q3 + 1.5 * IQR
    
        stock['point'][stock['point']> Max] = Max
        stock['point'][stock['point']< Min] = Min
    
    
        ##모델링
        scaler = MinMaxScaler()
        # 스케일을 적용할 column을 정의합니다.
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume','point']
        # 스케일 후 columns
        scaled = scaler.fit_transform(stock[scale_cols])
        df = pd.DataFrame(scaled, columns=scale_cols)
    
        x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.30, random_state=0, shuffle=False)
    
        def windowed_dataset(series, window_size, batch_size, shuffle):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size + 1))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.map(lambda w: (w[:-1], w[-1]))
            return ds.batch(batch_size).prefetch(1)
    
        WINDOW_SIZE=30
        BATCH_SIZE=50
    
        train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
        test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)
    
        model = Sequential([
        # 1차원 feature map 생성
        Conv1D(filters=32, kernel_size=5,padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
        # LSTM
        LSTM(16, activation='tanh'),Dense(16, activation="relu"), Dense(1),])
    
        loss = Huber()
        optimizer = Adam(0.0005)
        model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])
    
        # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
        # val_loss 기준 체크포인터도 생성합니다.
        filename = os.path.join('tmp', 'ckeckpointer.ckpt')
        checkpoint = ModelCheckpoint(filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)
    
        history = model.fit(train_data, validation_data=(test_data), epochs=100, callbacks=[checkpoint, earlystopping])
    
        model.load_weights(filename)
        
        
        def windowed_dataset1(series, window_size, batch_size, shuffle):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size , shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.map(lambda w: (w[:-1], w[-1]))
            return ds.batch(batch_size).prefetch(1)
    
        pred_data=df[['Close']][-30:]
        pred_data2 = windowed_dataset1(pred_data, WINDOW_SIZE, BATCH_SIZE, False)
        pred_data3=model.predict(pred_data2)
        
        rate=pred_data3[0][0] / df['Close'].iloc[-1] 
        rate_list.append(rate)
        
        #감성지수
        sentiment_list.append(sum(stock['point'][-66:])/66)
    
    
    korea_etf_risk.loc[:, '감성지수'] = sentiment_list
    
    korea_etf_risk.loc[:,'수익률'] = rate_list
    def growth(x) :
        if x > 1.0 :
            return '상승'
        elif x ==1 :
            return '유지'
        else :
            return '하락'
        
    def updown(x) :
        if x >= 0 :
            return '긍정'
        else:
            return '부정'
        
    
    korea_etf_risk['상승여부'] = korea_etf_risk['수익률'].apply(lambda x : growth(x))
    korea_etf_risk['감성지수'] = korea_etf_risk['감성지수'].apply(lambda x : updown(x))
    korea_etf_top3 = korea_etf_risk.sort_values(by =['수익률'],ascending = False).head(3)

    return(korea_etf_top3)

def global_top_etf(risk) :
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    import os
    !pip install finance-datareader
    import FinanceDataReader as fdr

    global_etf=pd.read_csv('해외ETF_상위_150.csv')
    global_etf['Ticker'] = global_etf['Ticker'].str.strip()
    global_etf_risk = global_etf[global_etf['level']==risk]

    
    def windowed_dataset(series, window_size, batch_size, shuffle):
        series = tf.expand_dims(series, axis=-1)
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size + 1))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.map(lambda w: (w[:-1], w[-1]))
        return ds.batch(batch_size).prefetch(1)

    rate_list =[]

    
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
    from tensorflow.keras.losses import Huber
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    
    

    for i in global_etf_risk['Ticker'] :
        stock=fdr.DataReader(str(i),'2018-08-01', '2021-08-01')
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        # 스케일을 적용할 column을 정의합니다.
        scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        # 스케일 후 columns
        scaled = scaler.fit_transform(stock[scale_cols])
        df = pd.DataFrame(scaled, columns=scale_cols)

        from sklearn.model_selection import train_test_split
        import tensorflow as tf
        x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.30, random_state=0, shuffle=False)

        WINDOW_SIZE=30
        BATCH_SIZE=30

        train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
        test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

        
    
        model = Sequential([
          # 1차원 feature map 생성
          Conv1D(filters=32, kernel_size=5, padding="causal", activation="relu", input_shape=[WINDOW_SIZE, 1]),
          # LSTM
          LSTM(16, activation='tanh'), Dense(16, activation="relu"), Dense(1),])

        loss = Huber()
        optimizer = Adam(0.0005)
        model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

        # earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
        earlystopping = EarlyStopping(monitor='val_loss', patience=10)
        # val_loss 기준 체크포인터도 생성합니다.
        filename = os.path.join('tmp', 'ckeckpointer.ckpt')
        checkpoint = ModelCheckpoint(filename, save_weights_only=True, save_best_only=True, monitor='val_loss', verbose=1)

        history = model.fit(train_data, validation_data=(test_data), epochs=50, callbacks=[checkpoint, earlystopping])
        model.load_weights(filename)
        def windowed_dataset1(series, window_size, batch_size, shuffle):
            series = tf.expand_dims(series, axis=-1)
            ds = tf.data.Dataset.from_tensor_slices(series)
            ds = ds.window(window_size , shift=1, drop_remainder=True)
            ds = ds.flat_map(lambda w: w.batch(window_size))
            if shuffle:
                ds = ds.shuffle(1000)
            ds = ds.map(lambda w: (w[:-1], w[-1]))
            return ds.batch(batch_size).prefetch(1)

        pred_data = df[['Close']][-30:]
        pred_data2 = windowed_dataset1(pred_data, WINDOW_SIZE, BATCH_SIZE, False)
        pred_data3 = model.predict(pred_data2)
        
        
        rate=pred_data3[0][0] / df['Close'].iloc[-1] 
        rate_list.append(rate)
        
    
    global_etf_risk.loc[:,'수익률'] = rate_list
    def growth(x) :
        if x > 1.0 :
            return '상승'
        elif x ==1 :
            return '유지'
        else :
            return '하락'
    
    global_etf_risk['상승여부'] = global_etf_risk['수익률'].apply(lambda x : growth(x))
    global_etf_top3 = global_etf_risk.sort_values(by =['수익률'],ascending = False).head(3)

    return(global_etf_top3)

kor_data_1 = korea_top_etf(1)
kor_data_2 = korea_top_etf(2)
kor_data_3 = korea_top_etf(3)

from firebase_admin import credentials
from firebase_admin import db

kor_dir_1_top1 = db.reference('recommend/1/koreatop1')
kor_dir_1_top2 = db.reference('recommend/1/koreatop2')
kor_dir_1_top3 = db.reference('recommend/1/koreatop3')
kor_dir_2_top1 = db.reference('recommend/2/koreatop1')
kor_dir_2_top2 = db.reference('recommend/2/koreatop2')
kor_dir_2_top3 = db.reference('recommend/2/koreatop3')
kor_dir_3_top1 = db.reference('recommend/3/koreatop1')
kor_dir_3_top2 = db.reference('recommend/3/koreatop2')
kor_dir_3_top3 = db.reference('recommend/3/koreatop3')

kor_dir_1_top1.update({'name' : str(kor_data_1.종목명.iloc[0])})
kor_dir_1_top1.update({'number' : str(kor_data_1.투자위험등급.iloc[0])})
kor_dir_1_top1.update({'tema' : str(kor_data_1.테마.iloc[0])})
kor_dir_1_top1.update({'rate' : str(round(kor_data_1.수익률.iloc[0]*100-100, 2))})
kor_dir_1_top1.update({'up' : str(kor_data_1.상승여부.iloc[0])})
kor_dir_1_top1.update({'sentiment' : str(kor_data_1.감성지수.iloc[0])})

kor_dir_1_top2.update({'name' : str(kor_data_1.종목명.iloc[1])})
kor_dir_1_top2.update({'number' : str(kor_data_1.투자위험등급.iloc[1])})
kor_dir_1_top2.update({'tema' : str(kor_data_1.테마.iloc[1])})
kor_dir_1_top2.update({'rate' : str(round(kor_data_1.수익률.iloc[1]*100-100, 2))})
kor_dir_1_top2.update({'up' : str(kor_data_1.상승여부.iloc[1])})
kor_dir_1_top2.update({'sentiment' : str(kor_data_1.감성지수.iloc[1])})

kor_dir_1_top3.update({'name' : str(kor_data_1.종목명.iloc[2])})
kor_dir_1_top3.update({'number' : str(kor_data_1.투자위험등급.iloc[2])})
kor_dir_1_top3.update({'tema' : str(kor_data_1.테마.iloc[2])})
kor_dir_1_top3.update({'rate' : str(round(kor_data_1.수익률.iloc[2]*100-100, 2))})
kor_dir_1_top3.update({'up' : str(kor_data_1.상승여부.iloc[2])})
kor_dir_1_top3.update({'sentiment' : str(kor_data_1.감성지수.iloc[2])})

kor_dir_2_top1.update({'name' : str(kor_data_2.종목명.iloc[0])})
kor_dir_2_top1.update({'number' : str(kor_data_2.투자위험등급.iloc[0])})
kor_dir_2_top1.update({'tema' : str(kor_data_2.테마.iloc[0])})
kor_dir_2_top1.update({'rate' : str(round(kor_data_2.수익률.iloc[0]*100-100, 2))})
kor_dir_2_top1.update({'up' : str(kor_data_2.상승여부.iloc[0])})
kor_dir_2_top1.update({'sentiment' : str(kor_data_2.감성지수.iloc[0])})

kor_dir_2_top2.update({'name' : str(kor_data_2.종목명.iloc[1])})
kor_dir_2_top2.update({'number' : str(kor_data_2.투자위험등급.iloc[1])})
kor_dir_2_top2.update({'tema' : str(kor_data_2.테마.iloc[1])})
kor_dir_2_top2.update({'rate' : str(round(kor_data_2.수익률.iloc[1]*100-100, 2))})
kor_dir_2_top2.update({'up' : str(kor_data_2.상승여부.iloc[1])})
kor_dir_2_top2.update({'sentiment' : str(kor_data_2.감성지수.iloc[1])})

kor_dir_2_top3.update({'name' : str(kor_data_2.종목명.iloc[2])})
kor_dir_2_top3.update({'number' : str(kor_data_2.투자위험등급.iloc[2])})
kor_dir_2_top3.update({'tema' : str(kor_data_2.테마.iloc[2])})
kor_dir_2_top3.update({'rate' : str(round(kor_data_2.수익률.iloc[2]*100-100, 2))})
kor_dir_2_top3.update({'up' : str(kor_data_2.상승여부.iloc[2])})
kor_dir_2_top3.update({'sentiment' : str(kor_data_2.감성지수.iloc[2])})

kor_dir_3_top1.update({'name' : str(kor_data_3.종목명.iloc[0])})
kor_dir_3_top1.update({'number' : str(kor_data_3.투자위험등급.iloc[0])})
kor_dir_3_top1.update({'tema' : str(kor_data_3.테마.iloc[0])})
kor_dir_3_top1.update({'rate' : str(round(kor_data_3.수익률.iloc[0]*100-100, 2))})
kor_dir_3_top1.update({'up' : str(kor_data_3.상승여부.iloc[0])})
kor_dir_3_top1.update({'sentiment' : str(kor_data_3.감성지수.iloc[0])})

kor_dir_3_top2.update({'name' : str(kor_data_3.종목명.iloc[1])})
kor_dir_3_top2.update({'number' : str(kor_data_3.투자위험등급.iloc[1])})
kor_dir_3_top2.update({'tema' : str(kor_data_3.테마.iloc[1])})
kor_dir_3_top2.update({'rate' : str(round(kor_data_3.수익률.iloc[1]*100-100, 2))})
kor_dir_3_top2.update({'up' : str(kor_data_3.상승여부.iloc[1])})
kor_dir_3_top2.update({'sentiment' : str(kor_data_3.감성지수.iloc[1])})

kor_dir_3_top3.update({'name' : str(kor_data_3.종목명.iloc[2])})
kor_dir_3_top3.update({'number' : str(kor_data_3.투자위험등급.iloc[2])})
kor_dir_3_top3.update({'tema' : str(kor_data_3.테마.iloc[2])})
kor_dir_3_top3.update({'rate' : str(round(kor_data_3.수익률.iloc[2]*100-100, 2))})
kor_dir_3_top3.update({'up' : str(kor_data_3.상승여부.iloc[2])})
kor_dir_3_top3.update({'sentiment' : str(kor_data_3.감성지수.iloc[2])})

data_1 = global_top_etf(1)
data_2 = global_top_etf(2)
data_3 = global_top_etf(3)

dir_1_top1 = db.reference('recommend/1/externaltop1')
dir_1_top2 = db.reference('recommend/1/externaltop2')
dir_1_top3 = db.reference('recommend/1/externaltop3')
dir_2_top1 = db.reference('recommend/2/externaltop1')
dir_2_top2 = db.reference('recommend/2/externaltop2')
dir_2_top3 = db.reference('recommend/2/externaltop3')
dir_3_top1 = db.reference('recommend/3/externaltop1')
dir_3_top2 = db.reference('recommend/3/externaltop2')
dir_3_top3 = db.reference('recommend/3/externaltop3')

dir_1_top1.update({'name' : str(data_1.Name.iloc[0])})
dir_1_top1.update({'number' : str(data_1.level.iloc[0])})
dir_1_top1.update({'rate' : str(round(data_1.수익률.iloc[0]*100-100, 2))})
dir_1_top1.update({'up' : str(data_1.상승여부.iloc[0])})

dir_1_top2.update({'name' : str(data_1.Name.iloc[1])})
dir_1_top2.update({'number' : str(data_1.level.iloc[1])})
dir_1_top2.update({'rate' : str(round(data_1.수익률.iloc[1]*100-100, 2))})
dir_1_top2.update({'up' : str(data_1.상승여부.iloc[1])})

dir_1_top3.update({'name' : str(data_1.Name.iloc[2])})
dir_1_top3.update({'number' : str(data_1.level.iloc[2])})
dir_1_top3.update({'rate' : str(round(data_1.수익률.iloc[2]*100-100, 2))})
dir_1_top3.update({'up' : str(data_1.상승여부.iloc[2])})

dir_2_top1.update({'name' : str(data_2.Name.iloc[0])})
dir_2_top1.update({'number' : str(data_2.level.iloc[0])})
dir_2_top1.update({'rate' : str(round(data_2.수익률.iloc[0]*100-100, 2))})
dir_2_top1.update({'up' : str(data_2.상승여부.iloc[0])})

dir_2_top2.update({'name' : str(data_2.Name.iloc[1])})
dir_2_top2.update({'number' : str(data_2.level.iloc[1])})
dir_2_top2.update({'rate' : str(round(data_2.수익률.iloc[1]*100-100, 2))})
dir_2_top2.update({'up' : str(data_2.상승여부.iloc[1])})

dir_2_top3.update({'name' : str(data_2.Name.iloc[2])})
dir_2_top3.update({'number' : str(data_2.level.iloc[2])})
dir_2_top3.update({'rate' : str(round(data_2.수익률.iloc[2]*100-100, 2))})
dir_2_top3.update({'up' : str(data_2.상승여부.iloc[2])})

dir_3_top1.update({'name' : str(data_3.Name.iloc[0])})
dir_3_top1.update({'number' : str(data_3.level.iloc[0])})
dir_3_top1.update({'rate' : str(round(data_3.수익률.iloc[0]*100-100, 2))})
dir_3_top1.update({'up' : str(data_3.상승여부.iloc[0])})

dir_3_top2.update({'name' : str(data_3.Name.iloc[1])})
dir_3_top2.update({'number' : str(data_3.level.iloc[1])})
dir_3_top2.update({'rate' : str(round(data_3.수익률.iloc[1]*100-100, 2))})
dir_3_top2.update({'up' : str(data_3.상승여부.iloc[1])})

dir_3_top3.update({'name' : str(data_3.Name.iloc[2])})
dir_3_top3.update({'number' : str(data_3.level.iloc[2])})
dir_3_top3.update({'rate' : str(round(data_3.수익률.iloc[2]*100-100, 2))})
dir_3_top3.update({'up' : str(data_3.상승여부.iloc[2])})

import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import openpyxl


dir = db.reference('news/title')
dor = db.reference('news/href')

baseUrl = 'https://search.naver.com/search.naver?where=news&query='
plusUrl = '경제'
baseUrlTail = '&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all&start='
url = baseUrl + urllib.parse.quote_plus(plusUrl) + baseUrlTail


count = 0

num = 1
newurl = url + str(num)
html = urllib.request.urlopen(newurl).read()
soup = BeautifulSoup(html, 'html.parser')
title = soup.find_all(class_='news_tit')
href = soup.find_all("a")
for j in title: 
    dir.update({count : str(j.attrs['title'])})
    dor.update({count : str(j.attrs['href']).replace("\"", "")})
    count += 1

kospi_dir = db.reference('king/kospi')

basic_url = "https://finance.naver.com/sise/sise_index.naver?code=KOSPI"

fp = urllib.request.urlopen(basic_url)

source = fp.read()

fp.close()

soup = BeautifulSoup(source, 'html.parser')
soup1 = soup.findAll("div","quotient dn")
soup2 = soup.find_all("span","fluc")


kospi_price = soup1[0].text
kospi_price = (kospi_price.split('\n'))[1]
kospi_up = str(soup2[0].text)
kospi_up = ((kospi_up.split("%"))[0]+'%').split(' ')
kospi_up = kospi_up[0] + '(' + kospi_up[1] + ')'
#print(soup)

kospi_dir.update({'price': str(kospi_price)})
kospi_dir.update({'up' : kospi_up})

nasdaq_dir = db.reference('king/nasdaq')

basic_url = "https://finance.naver.com/world/sise.naver?symbol=NAS@IXIC"

fp = urllib.request.urlopen(basic_url)

source = fp.read()

fp.close()

soup = BeautifulSoup(source, 'html.parser')
soup1 = soup.findAll("p","no_today")
soup2 = soup.find_all("p","no_exday")


nasdaq_price = soup1[0].text
nasdaq_price = (nasdaq_price.split('\n'))[2]
nasdaq_up = str(soup2[0].text).split('\n')
nasdaq_up = nasdaq_up[4] + nasdaq_up[7] + nasdaq_up[8] + nasdaq_up[9]
#print(soup)


nasdaq_dir.update({'price': str(nasdaq_price)})
nasdaq_dir.update({'up' : nasdaq_up})

sp500_dir = db.reference('king/s%p500')

basic_url = "https://finance.naver.com/world/sise.naver?symbol=SPI@SPX"

fp = urllib.request.urlopen(basic_url)

source = fp.read()

fp.close()

soup = BeautifulSoup(source, 'html.parser')
soup1 = soup.findAll("p","no_today")
soup2 = soup.find_all("p","no_exday")


sp_price = soup1[0].text
sp_price = (sp_price.split('\n'))[2]
sp_up = str(soup2[0].text).split('\n')
sp_up = sp_up[4] + sp_up[7] + sp_up[8] + sp_up[9]
#print(soup)


sp500_dir.update({'price': str(sp_price)})
sp500_dir.update({'up' : sp_up})

dollar_dir = db.reference('king/dollar')

basic_url = "https://finance.naver.com/marketindex/exchangeDetail.naver?marketindexCd=FX_USDKRW"

fp = urllib.request.urlopen(basic_url)

source = fp.read()

fp.close()

soup = BeautifulSoup(source, 'html.parser')
soup1 = soup.findAll("p","no_today")
soup2 = soup.find_all("p","no_exday")


dollar_price = soup1[0].text
dollar_price = (dollar_price.split('\n'))[3]
dollar_up = str(soup2[0].text).split('\n')
dollar_up = dollar_up[4] + dollar_up[7] + dollar_up[8] + dollar_up[9]
#print(soup)

dollar_dir.update({'price': str(dollar_price)})
dollar_dir.update({'up' : dollar_up})