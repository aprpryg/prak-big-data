# Ini adalah tugas kelompok 9

import yfinance as yf
import pandas as pd
import numpy as np
import math
import streamlit as st
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
# define the ticker symbol in a dictionary, with its corresponding full name
ticker_dict = {
    'ANTM.JK': "PT Aneka Tambang Tbk",
    'BMRI.JK': "PT Bank Mandiri (Persero) Tbk",
    'BBNI.JK': "PT Bank Negara Indonesia (Persero) Tbk",
    'PNBN.JK': "PT Bank Pan Indonesia Tbk",
    'ISAT.JK': "PT Indosat Tbk",
    'JSMR.JK': "PT Jasa Marga (Persero) Tbk",
    'LPGI.JK': "PT Lippo General Insurance Tbk",
    'FREN.JK': "PT Smartfren Telecom Tbk",
    'TLKM.JK': "PT Telekomunikasi Indonesia Tbk",
    'EXCL.JK': "PT XL Axiata Tbk",
    'GOOGL': "Google",
    'MSFT': "Microsoft",
    'AAPL': "Apple",
    'META': "Facebook Meta"
}

st.write("""
# Aplikasi Yahoo Finance

## Data Saham Beberapa Perusahaan

""")

tickerSymbols = sorted(ticker_dict.keys())

ticker = st.selectbox(
    "Ticker Perusahaan",
    options = tickerSymbols
)

st.write(f'Ticker perusahaan: **{ticker_dict[ticker]}**')

tickerData = yf.Ticker(ticker)

hari_mundur = st.selectbox(
    "Pilihan rentang hari",
    options = [7, 10, 20, 30, 60, 90, 365]
)

jumlah_hari = timedelta(days = -int(hari_mundur))

tgl_mulai = date.today() + jumlah_hari

tgl_akhir = st.date_input(
    "Hingga",
    value=date.today()
)

tickerDF = tickerData.history(
    period='1d',
    start=str(tgl_mulai),
    end=str(tgl_akhir)
)

attributes = st.multiselect(
    "Informasi yang ditampilkan:",
    options=['Open', 'High', 'Low', 'Close', 'Volume'],
    default=['Open', 'Close']
)

st.markdown(f"Lima data pertama:")
st.write(tickerDF.head())

#st.plotly_chart( px.line(tickerDF.Open) )
#st.plotly_chart( px.line(tickerDF["High"]) )

judul_chart = f'Harga Saham {ticker_dict[ticker]} ({ticker})'

from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regression = LinearRegression()
regression.fit(train_x, train_y)
print(f"{regression.coef_ = }")
print(f"{regression.intercept_ = }")

predicted=regression.predict(test_x)
print(test_x.head())

dfr=pd.DataFrame({'Actual_Price':test_y, 'Predicted_Price':predicted})
dfr.head(10)

print('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(test_y, predicted))
print('Mean Squared Error (MSE) :',
      metrics.mean_squared_error(test_y, predicted))
print('Root Mean Squared Error (RMSE):',
      np.sqrt(metrics.mean_squared_error(test_y, predicted)))

x2 = dfr.Actual_Price.mean()
y2 = dfr.Predicted_Price.mean()
Accuracy1 = x2/y2*100
print("The accuracy of the model is " , Accuracy1) 

st.plotly_chart(
    px.line(
        tickerDF,
        title=judul_chart,
        y = attributes
    )  
)

history = tickerData.history(period="Max")
df = pd.DataFrame(history)
df.head(10)

x = df[['Open', 'High','Low', 'Volume']]
y = df['Close']

# Linear regression Model for stock prediction 
train_x, test_x, train_y, test_y = train_test_split(x, y,
                                                    test_size=0.15,
                                                    shuffle=False,
                                                    random_state = 0)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
regression = LinearRegression()
regression.fit(train_x, train_y)
st.write(f"{regression.coef_ = }")
st.write(f"{regression.intercept_ = }")

predicted=regression.predict(test_x)
print(test_x.head())

st.write('Mean Absolute Error (MAE):',
      metrics.mean_absolute_error(test_y, predicted))
st.write('Mean Squared Error (MSE) :',
      metrics.mean_squared_error(test_y, predicted))
st.write('Root Mean Squared Error (RMSE):',
      np.sqrt(metrics.mean_squared_error(test_y, predicted)))

dfr=pd.DataFrame({'Actual_Price':test_y, 'Predicted_Price':predicted})
dfr.head(10)

x2 = dfr.Actual_Price.mean()
y2 = dfr.Predicted_Price.mean()
Accuracy1 = x2/y2*100
print("The accuracy of the model is " , Accuracy1)
