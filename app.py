from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

import math
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import json
import urllib.request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("index.html")
@app.route('/predict',methods=['POST','GET'])
def predict():
    val = request.form
    stock_name = val.get("Stock")
    print(stock_name)
    open_list = []
    high_list = []
    low_list = []
    close_list = []
    date_list = []
    adj_list = []
    volume_list = []
    test_size = 0.2                 # proportion of dataset to be used as test set
    cv_size = 0.2                   # proportion of dataset to be used as cross-validation set
    Nmax = 30                       # for feature at day t, we use lags from t-1, t-2, ..., t-N as features
                                    # Nmax is the maximum N we are going to test
    fontsize = 14
    ticklabelsize = 14

    def get_all_values(nested_dictionary):
        for key, value in nested_dictionary.items():
            if type(value) is dict:
                date_list[:0] = [key]
                get_all_values(value)
            else:
                if(key == "1. open"):
                    open_list[:0] = [float(value)]
                if(key == "2. high"):
                    high_list[:0] = [float(value)]
                if(key == "3. low"):
                    low_list[:0] = [float(value)]
                if(key == "4. close"):
                    close_list[:0] = [float(value)]
                if(key == "5. adjusted close"):
                    adj_list[:0] = [float(value)]
                if(key == "6. volume"):
                    volume_list[:0] = [float(value)]


    API_key = "8WD1G387BSGG76A8"
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=" + stock_name + "&apikey=" + API_key
    # url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=demo"
    data = urllib.request.urlopen(url).read().decode()
    obj = json.loads(data)
    obj.pop("Meta Data")
    obj["Time Series (Daily)"]
    get_all_values(obj["Time Series (Daily)"])

    def get_preds_lin_reg(df, target_col, N, pred_min, offset):
        
        # Create linear regression object
        regr = LinearRegression(fit_intercept=True)

        pred_list = []

        for i in range(offset, len(df['adj_close'])):
            X_train = np.array(range(len(df['adj_close'][i-N:i]))) # e.g. [0 1 2 3 4]
            y_train = np.array(df['adj_close'][i-N:i]) # e.g. [2944 3088 3226 3335 3436]
            X_train = X_train.reshape(-1, 1)     
            y_train = y_train.reshape(-1, 1)
    
            regr.fit(X_train, y_train)            # Train the model
            pred = regr.predict(np.array(N).reshape(1,-1))
        
            pred_list.append(pred[0][0])  # Predict the footfall using the model
        
        pred_list = np.array(pred_list)
        pred_list[pred_list < pred_min] = pred_min
            
        return pred_list

    def get_mape(y_true, y_pred): 
        
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    data = {"date":date_list,"open": open_list,"high":high_list,"low": low_list,"close":close_list
            ,"adj_close":adj_list,"volume":volume_list}
    data = pd.DataFrame(data)
    # data.index = date_list
    df = data
    df

    rcParams['figure.figsize'] = 13, 6 

    ax = df.plot(x='date', y='adj_close', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("USD")

    num_cv = int(cv_size*len(df))
    num_test = int(test_size*len(df))
    num_train = len(df) - num_cv - num_test
    print("num_train = " + str(num_train))
    print("num_cv = " + str(num_cv))
    print("num_test = " + str(num_test))

    # Split into train, cv, and test
    train = df[:num_train].copy()
    cv = df[num_train:num_train+num_cv].copy()
    train_cv = df[:num_train+num_cv].copy()
    test = df[num_train+num_cv:].copy()
    print("train.shape = " + str(train.shape))
    print("cv.shape = " + str(cv.shape))
    print("train_cv.shape = " + str(train_cv.shape))
    print("test.shape = " + str(test.shape))

    RMSE = []
    R2 = []
    mape = []
    for N in range(1, Nmax+1): # N is no. of samples to use to predict the next value
        est_list = get_preds_lin_reg(train_cv, 'adj_close', N, 0, num_train)
        
        cv.loc[:, 'est' + '_N' + str(N)] = est_list
        RMSE.append(math.sqrt(mean_squared_error(est_list, cv['adj_close'])))
        R2.append(r2_score(cv['adj_close'], est_list))
        mape.append(get_mape(cv['adj_close'], est_list))
    print('RMSE = ' + str(RMSE))
    print('R2 = ' + str(R2))
    print('MAPE = ' + str(mape))
    cv.head()

    matplotlib.rcParams.update({'font.size': 14})
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(range(1, Nmax+1), RMSE, 'x-')
    plt.grid()
    plt.xlabel('N')
    plt.ylabel('RMSE')
    plt.xlim([2, 30])

    # Set optimum N
    N_opt = 8

    #Final Model
    est_list = get_preds_lin_reg(df, 'adj_close', N_opt, 0, num_train+num_cv)
    est_list #Predicted value on testing data
    est_list = pd.DataFrame(est_list)
    # est_list
    est_list.index = test.index

    test['adj_close'].plot()
    est_list[0].plot()

    ####Final Today's Prediction

    from datetime import datetime

    now = datetime.now()
    day = now.strftime("%Y/%d/%m")
    Nmax2 = 8

    df_temp = cv[cv['date'] <= day]
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(range(1,Nmax2+2), df_temp[-Nmax2-1:]['adj_close'], 'bx-')
    plt.plot(Nmax2+1, df_temp[-1:]['adj_close'], 'ys-')
    legend_list = ['adj_close', 'actual_value']

    regr = LinearRegression(fit_intercept=True) # Create linear regression object
    # pickle.dump(regr,open("model.pkl","wb"))
    # model = pickle.load(open("model.pkl","rb"))
    for N in range(8, Nmax2+1):
        # Plot the linear regression lines
        X_train = np.array(range(len(df_temp['adj_close'][-N-1:-1]))) # e.g. [0 1 2 3 4]
        y_train = np.array(df_temp['adj_close'][-N-1:-1]) # e.g. [2944 3088 3226 3335 3436]
        X_train = X_train.reshape(-1, 1)     
        y_train = y_train.reshape(-1, 1)
        regr.fit(X_train, y_train)            # Train the model
        today_Pred = regr.predict(X_train)         # Get linear regression line
        print(today_Pred)
        a = str(today_Pred[7])
    s = "The Predicted Value of Stock for today is: " + a
    return s
if __name__ == '__main__':
    app.run(debug=True)

