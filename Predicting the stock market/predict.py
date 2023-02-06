import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error



stocks = pd.read_csv('sphist.csv')

stocks['Date'] = pd.to_datetime(stocks['Date'])

stocks.sort_values('Date', ascending=True, inplace=True)

stocks = stocks.reset_index()



for column in stocks.columns.drop(['Date', 'index']):
    stocks['rolling_'+ column] = stocks[column].rolling(30).mean()


columns = stocks.columns

columns = [col for col in columns if 'rolling' in col]


stocks[columns] = stocks[columns].shift()

stocks.dropna(inplace=True)

train = stocks[stocks['Date'] < datetime(2013, 1, 1)].copy()
test = stocks[stocks['Date'] >= datetime(2013, 1, 1)].copy()

lr = LinearRegression()

columns = [col for col in columns if 'Close' not in col]

lr.fit(train[columns[:3]], train['Close'])
predictions = lr.predict(test[columns[:3]])
error = mean_absolute_error(test['Close'], predictions)
error



if __name__ == '__main__':
	print(predictions)


