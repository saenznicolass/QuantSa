"""
@author: Nico
"""

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from scipy.stats import skew, kurtosis, chi2


# get market data from folder
directory = '../Data/' # hardcoded
rick = '^STOXX50E'


'''
Let's download data with Yahoo's API
'''
# Uncomment to dowload data with Yahoo's API
end_date = dt.date(2020,12,30)
start_date =  dt.date(2016,1,4)
#start_date = end_date - dt.timedelta(days=365*2) 
raw_data  = yf.download(rick, start_date, end_date) #ordered by default
tr = pd.DataFrame(index=raw_data.index) #create a table with ordered indexes

# # Comment to download data with Yahoo's API
# path = directory + rick + '.csv' 
# raw_data = pd.read_csv(path)
# tr = pd.DataFrame(index=(pd.to_datetime(raw_data['Date'], dayfirst=True))) # create a table with ordered indexes

# Building table of returns
tr['Close'] = raw_data.Close.values
tr.sort_values(by='Date', ascending=True)
tr['Close_previous'] = tr['Close'].shift(1)
tr['return_close'] = tr['Close']/tr['Close_previous'] - 1
tr = tr.dropna()


'''
Let's create a Jarque-Bera normality test
'''
x = tr['return_close'].values
x_description = 'market data ' + rick
n_obs = len(x) 

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurtosis = kurtosis(x) # 4th moment. Excess kurtosis. Tales
x_jb = n_obs/6*(x_skew**2 + 1/4*x_kurtosis**2)
x_p_value = 1 - chi2.cdf(x_jb, df=2) # normal is p>0.05. 2 to be distributed as chi-square
x_is_normal = (x_p_value > 0.05) # equivalently jb < 6

print('---Real market data---')
print('Rick is ' + rick)
print('mean is ' + str(x_mean))
print('standard deviation is ' + str(x_std))
print('skewness is ' + str(x_skew))
print('kurtosis is ' + str(x_kurtosis))
print('JB statistic is ' + str(x_jb))
print('p-value ' + str(x_p_value))
print('is normal ' + str(x_is_normal))

# plot timeseries of price
plt.figure()
plt.plot(tr['Close'])
plt.title('Time series real prices ' + rick)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

# plot histogram
plt.figure()
plt.hist(x,bins=100)
plt.title(x_description)
plt.show()





# ================Some comments to understand the code=========================
# Dayfirst = True: EU format DD/MM/YY
# Shift() Move days so we can compute returns 
# reset_index(drop=True): once nan values deleted, we have to reset indexes
# =============================================================================

