# -*- coding: utf-8 -*-
"""
Eric Born

"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.model_selection import train_test_split

# Using the stock ticker BSX - Boston Scientific Corporation.
# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\TomBrody\Desktop\Projects\stock-prediction\data'
ticker_file = os.path.join(input_dir, ticker + '.csv')

# Set display options for dataframes
#pd.set_option('display.max_rows', 100)
#pd.set_option('display.width', 500)
#pd.set_option('display.max_columns', 50)

# Set seaborn color palette and grid style
# sns.set_palette(sns.light_palette("Blue", reverse=False))
# GnBu_d
sns.set_palette(sns.color_palette("Blues_d"))
sns.set_style('darkgrid')

# read csv file into dataframe
try:
    df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', ticker,'\n')

except Exception as e:
    print(e)
    print('failed to read stock data for ticker: ', ticker)

# Create class column where red = 0 and green = 1
df['class'] = df['label'].apply(lambda x: 1 if x =='green' else 0)

# Create separate dataframes for 2017 and 2018 data
# 2017 will be used as training, 2018 as testing for the model
df_2017 = df.loc[df['td_year']==2017]
df_2018 = df.loc[df['td_year']==2018]

# Reset indexes
df_2017 = df_2017.reset_index(level=0, drop=True)
df_2018 = df_2018.reset_index(level=0, drop=True)

# Create reduced dataframe only containing week number, mu, sig and label
df_2017_reduced = pd.DataFrame( {'week nbr' : range(1, 53),
                'mu'    : df_2017.groupby('td_week_number')['return'].mean(),
                'sig'   : df_2017.groupby('td_week_number')['return'].std(),
                'label' : df_2017.groupby('td_week_number')['class'].first()})

# Create reduced dataframe only containing week number, mu, sig and label
df_2018_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
                'mu'    : df_2018.groupby('td_week_number')['return'].mean(),
                'sig'   : df_2018.groupby('td_week_number')['return'].std(),
                'label' : df_2018.groupby('td_week_number')['class'].first()})

# Replacing nan in week 52 sigma column with a zero due to 
# there being only 1 trading day that week.
df_2018_reduced = df_2018_reduced.fillna(0)

# remove index name labels from dataframes
del df_2017_reduced.index.name
del df_2018_reduced.index.name

# Define features labels
features = ['mu', 'sig']

#####
# Setup train/test and scale data
#####

# Initialize scaler
scaler = StandardScaler()

# Create x training and test sets from 2017/2018 features values.
x_train = df_2017_reduced[features].values
x_test = df_2018_reduced[features].values

# create y training and test sets from 2017/2018 label values
y_train = df_2017_reduced['label'].values
y_test = df_2018_reduced['label'].values

# Scaler for training data
scaler.fit(x_train)
x_2017_train = scaler.transform(x_train)

# Scaler for test data
scaler.fit(x_test)
x_2018_test = scaler.transform(x_test)

#####
# KNN
#####

# Create a KNN model based upon the mean and standard
# deviation measures of the weekly stock returns

# Create empty lists to store the models error rate and 
# accuracy across various K's
error_rate = []
accuracy = []
k_value = []

# For loop to train the model with 2017 data and test on 2018 data
# with k neighbors set to 3 to 10
try:
    for k in range (3, 11, 1):
        # Create the classifier with neighbors set to k from the loop
        knn_classifier = KNeighborsClassifier(n_neighbors = k)
       
        # Train the classifier
        knn_classifier.fit(x_train, y_train)
        
        # Perform predictions
        pred_k = knn_classifier.predict(x_test)
        
        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(pred_k != y_test) * 100, 2))
        accuracy.append(round(sum(pred_k == y_test) / len(pred_k) * 100, 2))
except Exception as e:
    print(e)
    print('failed to build the KNN classifier.')

for i in range (0,5):
    print('The accuracy on 2017 data when K =', k_value[i], 'is:', accuracy[i])
    
# create a plot to display the accuracy of the model across K
fig = plt.figure(figsize=(10, 4))
ax = plt.gca()
plt.plot(range(3, 11, 1), accuracy, color ='blue',
         marker = 'o', markerfacecolor = 'black', markersize = 10)
plt.title('Accuracy vs. k for stock labels')
plt.xlabel('Number of neighbors: k')
plt.ylabel('Accuracy')

# setup and test on 2018 data with k = 7
# Create x test set for 2018
x_test = df_2018_reduced[features].values
y_2018_test = df_2018_reduced['label'].values

# scaler for 2018 test data
scaler.fit(x_test)
x_2018_test = scaler.transform(x_test)

# Create the classifier with neighbors set to 5
knn_2018 = KNeighborsClassifier(n_neighbors = 7)

# Train the classifier using all of 2017 data
knn_2018.fit(x_train, y_train)
        
# Perform predictions on 2018 data
pred_2018 = knn_2018.predict(x_2018_test)

# Capture error and accuracy rates for 2018 predictions
error_2018 = round(np.mean(pred_2018 != y_2018_test) * 100, 2)
accuracy_2018 = round(sum(pred_2018 == y_2018_test) / len(pred_2018) * 100, 2)

# accuracy is 83.13%
print('\nThe accuracy on 2018 data when K = 5 is:', accuracy_2018, '%')

# Output the confusion matrix
cm = confusion_matrix(y_2018_test, pred_2018)
print('\nConfusion matrix for year 2 predictions:')
print(cm, '\n')

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BrBG", fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# what is true positive rate (sensitivity or recall) and true
# negative rate (specificity) for year 2?
print('The specificity is: 20/29 = 0.69 = 69%')	
print('The recall is: 23/24 =', 
      round(recall_score(y_2018_test, pred_2018) * 100, 2),'%')

# Implemented trading strategy based upon label predicitons vs
# buy and hold strategy

# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
worth = 0

# stores adj_close values for the last day of each trading week
adj_close = df_2018.groupby('td_week_number')['adj_close'].last()

# stores open price for the first day of each trading week
open_price = df_2018.groupby('td_week_number')['open'].first()

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for i in range(0, len(pred_2018)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if pred_2018[i] == 0 and shares > 0:
            wallet = round(shares * adj_close[i - 1], 2)
            shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if pred_2018[i] == 1 and shares == 0: 
            shares = wallet / open_price[i]
            wallet = 0            
            
except Exception as e:
    print(e)
    print('Failed to evaluate df_2018 labels')


# set worth by multiplying stock price on final day by total shares
worth = round(shares * adj_close[52], 2)

if worth == 0:
    worth = wallet
    profit = round(wallet - 100.00, 2)
else:
    profit = round(worth - 100.00, 2)

# Total Cash: $0
# Total shares: 6.703067 
# Worth: $236.89
# This method would close the year at $ 141.7 a profit of $ 41.7
print('\n2018 Label Strategy:')
print('Total Cash: $', wallet, '\nTotal shares:', round(shares, 6),
      '\nWorth: $', worth)    
print('This method would close the year at $', worth, 'a profit of $', profit)

# Buy and hold
# Initialize wallet and shares to track current money and number of shares.
wallet = 100.00
shares = 0
profit = 0
worth = 0

# Calculate shares, worth and profit
shares = round(wallet / float(open_price[0]), 6)
worth = round(shares * adj_close[52], 2)
profit = round(worth - 100.00, 2)

#Currently own 4.009623 shares 
#Worth $ 141.70
#Selling on the final day would result in $ 141.7 a profit of $ 41.7
print('\n2018 buy and hold:','\nCurrently own', shares, 'shares',
      '\nWorth','$',"%.2f"%round(worth, 2))
print('Selling on the final day would result in $',"%.2f"%worth, 
      'a profit of $', "%.2f"%profit)

###########
# This section of code calculates the regression for each day incrementing
# through a window size from 5 to 30 days.
# If the predicted close price for w+1 is greater than the close price for w,
# a 1 is put into the position column in the df_2017 dataframe.

# If the predicted close price for w+1 is less than the close price for w
# a -1 is put into the position column in the df_2017 dataframe.

# If the predicted close price for w+1 is equal to the close price for w
# a 0 is put into the position column in the df_2017 dataframe.

# window = number of days to evalute before making a prediction values 5-30
# adj_close price is used to train the regression model
# close price is being predicted
# window_end = window - 1 = total size of the window
# window_start = start of the window
# adj_close = array of adj_close prices inside window (x axis)
# close = array of close prices inside window (y axis)

'''
Takes x and y as inputs and calculates the size of the array,
the sum of squares, slope and intercept.
'''
def estimate_coef (x, y):
    n = np.size(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y * x) - n * mu_y * mu_x
    SS_xx = np.sum(x * x) - n * mu_x * mu_x
    slope = SS_xy / SS_xx
    intercept = mu_y - slope * mu_x
    return (slope, intercept)

# Creates a linear regression object
lm = LinearRegression()

# stores the position 0, 1, -1 for each window size
position_2017_df  = pd.DataFrame()

# list that will be populated from the below for loop
# to contains the window sizes
window_size = []

try:
    for window in range(5,31):
        # Create position column to indicate whether its a buy or sell day.
        # column is reset to all 0's at the start of each loop iteration
        df_2017['position'] = 0
        df_2017['prediction'] = 0
        
        # window size list populated with size increments
        window_size.append(window)
        
        # set window_end equal to window - 1 due to zero index
        window_start = 0
        window_end = window - 1
        
        # loop that handles gathering the adj_close and close price 
        # for the appropriate window size
        for rows in range(0, len(df_2017)):
            adj_close = np.array(df_2017.loc[window_start:window_end,
                                     'adj_close']).reshape(-1, 1)
            close = np.array(df_2017.loc[window_start:window_end,
                                     'close']).reshape(-1, 1)
            lm.fit(adj_close, close)
            
            # Breaks on the last row since it cannot predict w + 1 if 
            # there is no data for the next day, else it creates
            # a prediction.
            if window_end == len(df_2017) - 1:
                break
            else:
                pred = lm.predict(np.array(df_2017.loc[window_end + 1, 
                                          'adj_close']).reshape(-1, 1))
            
            # store the predicted value in the 2018 dataframe
            df_2017.loc[window_end + 1, 'prediction'] = float(pred) 
    
            # updates the position column with a 1 when prediciton for tomorrows
            # close price (w + 1) is greater than the close price of w.
            # Else it marks it with a -1 to indicate a lower price.
            if float(pred) > df_2017.loc[window_end, 'close']:
                df_2017.loc[window_end, 'position'] = 1
            elif float(pred) == df_2017.loc[window_end, 'close']:
                df_2017.loc[window_end, 'position'] = 0
            else:
                df_2017.loc[window_end, 'position'] = -1
            window_start += 1
            window_end += 1
        
        # writes the position column to a the position dataframe after each
        # window iteration
        position_2017_df[str(window)] = df_2017.loc[:, 'position']

except Exception as e:
    print(e)
    sys.exit('Failed to perform predictions on 2017 data')
    
# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
long_shares = 0
long_worth = 0
long_price = 0
name_increment = 5
trade_data_2017_df = pd.DataFrame()

# Manual variable setters for testing
#position_df.iloc[:, 0]
#position_column = 0
#position_row = 4

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_2017_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        long_price_name  = 'long_price'  + str(name_increment)
        long_shares_name = 'long_shares' + str(name_increment)
        long_worth_name  = 'long_worth'  + str(name_increment)

        for position_row in range(0, len(position_2017_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_2017_df.iloc[position_row, position_column] == 1 
            and long_shares == 0): 
                long_shares = 100.00 / df_2017.loc[position_row, 'close']           
                long_price = df_2017.loc[position_row, 'close']
                trade_data_2017_df.at[position_row, long_price_name] = long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_2017_df.iloc[position_row, position_column] == -1
            and long_shares != 0): 
                long_worth = ((long_shares 
                              * df_2017.loc[position_row, 'close'])
                              - long_price * long_shares)
                trade_data_2017_df.at[position_row, long_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_2017_df.at[position_row, long_price_name] = (
                                            df_2017.loc[position_row, 'close'])
                long_shares = 0
                long_price = 0
                long_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_2017_df.at[position_row, long_shares_name]  = long_shares
       
            # Manual increments for testing
            #position_column += 1
            #position_row += 1

        # increments the name_increment to represent the window size
        name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('Failed to build trading data for trade_data_df')            

# NaN are excluded when using the mean function so I decided to leave them in
# Replace all NaN with 0's
#trade_data_df = trade_data_df.fillna(0)            

# export trade data to CSV
#try:  
#    trade_data_df.to_csv(r'C:\Users\TomBrody\Desktop\School\677\wk4\trade_data.csv', index = False)
#
#except Exception as e:
#    print(e)
#    sys.exit('failed to export trade_data_df to csv')     

# sample data selections
#trade_data_df.iloc[0:10, 0:3]
#trade_data_df.iloc[0:10, 77]
#trade_data_df.iloc[0:3, 2]
#trade_data_df.iloc[:, column]

# creates a list containing the column names from the trade_data_df
name_list = []
for column in range(2, len(trade_data_2017_df.iloc[0, :]), 3):
    name_list.append(trade_data_2017_df.iloc[:, column].name)

# create a dataframe to store the daily profits made from selling stocks
summary_2017_df = trade_data_2017_df[name_list].copy()

# Sum of profits by window size
profit_2017 = summary_2017_df.sum()
print(profit_2017)

# creates a barplot of the window size vs the sum of profits in dollars
sns.barplot(window_size, summary_2017_df.sum(), palette = 'Blues_d')
plt.tight_layout()
plt.title('Window Size vs. Total Return')
plt.xlabel('Window Size')
plt.ylabel('Total Return in Dollars')
plt.show()

# mean of profits by window size
mean_2017 = summary_2017_df.mean()
print(mean_2017)

# creates a barplot of the window size vs the average return in dollars
sns.lineplot(window_size, summary_2017_df.mean(), color = 'navy')
plt.tight_layout()
plt.title('Window Size vs. Average Return')
plt.xlabel('Window Size')
plt.ylabel('Avg Return in Dollars')
plt.show()

# creates a barplot of the window size vs the average return in dollars
sns.lineplot(window_size, summary_2017_df.sum(), color = 'navy')
plt.tight_layout()
plt.title('Window Size vs. Total Return')
plt.xlabel('Window Size')
plt.ylabel('Total Return in Dollars')
plt.show()

# Review the number of trades by window size
trades_2017 = summary_2017_df.count()
print(trades_2017)

# creates a barplot of the number of trades by window size
sns.barplot(window_size, summary_2017_df.count(), palette = 'Blues_d')
plt.tight_layout()
plt.title('Window Size vs. Number of Trades')
plt.xlabel('Window Size')
plt.ylabel('Total Number of Trades')
plt.show()


###### Part 2
# Using a fixed window size of 5 which was determined in part 1 above
# to analyze and predict stock prices in data from 2018.
# Unfortunately the two measures being used, closing and adjusted 
# closing price are exactly the same every day in 2018
# so the results are a bit uninteresting

# Setup column in the 2018 dataframe to track the buy/sell position,
# the predicted value and initial start and end window positions
df_2018['position'] = 0
df_2018['prediction'] = 0
window_start = 0
window_end = 4

# stores the position 0, 1, -1 for each window size
position_2018_df  = pd.DataFrame()
try:    
    # loop that handles gathering the adj_close and close price 
    # for the 2018 dataframe
    for rows in range(0, len(df_2018)):
        adj_close = np.array(df_2018.loc[window_start:window_end,
                                 'adj_close']).reshape(-1, 1)
        close = np.array(df_2018.loc[window_start:window_end,
                                 'close']).reshape(-1, 1)
        # fits the model using adjusted close and close stock prices
        lm.fit(adj_close, close)
        
        # Breaks on the last row since it cannot predict w + 1 if 
        # there is no data for the next day, else it creates
        # a prediction.
        if window_end == len(df_2018) - 1:
            break
        else:
            pred = lm.predict(np.array(df_2018.loc[window_end + 1, 
                                      'close']).reshape(-1, 1))
        
        # store the predicted value in the 2018 dataframe
        df_2018.loc[window_end + 1, 'prediction'] = float(pred) 
        
        # updates the position column with a 1 when prediciton for tomorrows
        # close price (w + 1) is greater than the close price of w.
        # Else it marks it with a -1 to indicate a lower price.
        if float(pred) > df_2018.loc[window_end, 'close']:
            df_2018.loc[window_end, 'position'] = 1
        elif float(pred) == df_2018.loc[window_end, 'close']:
            df_2018.loc[window_end, 'position'] = 0
        else:
            df_2018.loc[window_end, 'position'] = -1
        window_start += 1
        window_end += 1

        # writes the position column to a the position dataframe after each
        # window iteration
        position_2018_df[str(window)] = df_2018.loc[:, 'position']

except Exception as e:
    print(e)
    sys.exit('Failed to build prediction data for 2018')       

# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
long_shares = 0
long_worth = 0
long_price = 0
name_increment = 5
trade_data_2018_df = pd.DataFrame()

# Manual variable setters for testing
#position_2018_df.iloc[:, 0]
#position_column = 0
#position_row = 4

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_2018_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        long_price_name  = 'long_price'  + str(name_increment)
        long_shares_name = 'long_shares' + str(name_increment)
        long_worth_name  = 'long_worth'  + str(name_increment)

        for position_row in range(0, len(position_2018_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_2018_df.iloc[position_row, position_column] == 1 
            and long_shares == 0): 
                long_shares = 100.00 / df_2018.loc[position_row, 'close']           
                long_price = df_2018.loc[position_row, 'close']
                trade_data_2018_df.at[position_row, long_price_name] = long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_2018_df.iloc[position_row, position_column] == -1
            and long_shares != 0): 
                long_worth = ((long_shares 
                              * df_2018.loc[position_row, 'close'])
                              - long_price * long_shares)
                trade_data_2018_df.at[position_row, long_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_2018_df.at[position_row, long_price_name] = (
                                            df_2018.loc[position_row, 'close'])
                long_shares = 0
                long_price = 0
                long_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_2018_df.at[position_row, long_shares_name]  = long_shares
       
            # Manual increments for testing
            #position_column += 1
            #position_row += 1

        # increments the name_increment to represent the window size
        name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('Failed to build trading data for trade_data_df')            

# Finish data building
#########
# Start analysis/presenation
    
# create a dataframe to store the daily profits made from selling stocks
summary_2018_df = trade_data_2018_df['long_worth5'].copy()

# Creating an estimated coefficient between the measures
# the slope is a perfect 1 and the intercept is at 0
# dataframes start at position 5 since the first 0-4 were not predicted
# do to window size starting at 5
coefficient = estimate_coef(df_2018.loc[5:,'adj_close'], df_2018.loc[5:,'prediction'])
print(coefficient)

# Generate a plot of the actual vs predicted values
sns.scatterplot(df_2018.loc[5:,'adj_close'], df_2018.loc[5:,'prediction'], color='navy')
sns.lineplot(range(25, 40), range(25, 40), color = 'red')
plt.title('Actual Close vs. Predicted Close')
plt.xlabel('Actual Close')
plt.ylabel('Predicted Close')
plt.show()

# average R2 value
# 0.5
print(np.mean(coefficient))

# long days
# 131
long_days = df_2018[df_2018['position'] > 0].count()['position']
print(long_days)

# short days
# 111
short_days = df_2018[df_2018['position'] < 0].count()['position']
print(short_days)

# Plot long vs short day totals
plot_data = pd.DataFrame({'Type': ['Long', 'Short'], 
                          'Days': [int(long_days), int(short_days)]})
sns.barplot(x = 'Type', y = 'Days', data = plot_data) 
plt.title('Long Days vs. Short Days')
plt.xlabel('Type')
plt.ylabel('Total Days')
plt.show()

# Sum of profits for 2018
profit_2018 = summary_2018_df.sum()
print(profit_2018)

# Plot total profits 2017 vs 2018
profit_data = pd.DataFrame({'Year': ['2017', '2018'], 
                            'Profit': [int(profit_2017[0]), int(profit_2018)]})
sns.barplot(x = 'Year', y = 'Profit', data = profit_data) 
plt.title('Total Profit 2017 vs. 2018')
plt.xlabel('Year')
plt.ylabel('Total Profit in Dollars')
plt.show()

# mean of profits
mean_2018 = round(summary_2018_df.mean(), 2)
print(mean_2018)

# Plot mean trade amount 2017 vs 2018
mean_data = pd.DataFrame({'Year': ['2017', '2018'], 
                            'Average': [float(mean_2017[0]), float(mean_2018)]})
sns.barplot(x = 'Year', y = 'Average', data = mean_data) 
plt.title('Average Profit 2017 vs. 2018')
plt.xlabel('Year')
plt.ylabel('Average Profit in Dollars')
plt.show()

# Review the number of trades by window size
trades_2018 = summary_2018_df.count()
print(trades_2018)

# Plot total number of trades 2017 vs 2018
trade_data = pd.DataFrame({'Year': ['2017', '2018'], 
                            'Trades': [int(trades_2017[0]), int(trades_2018)]})
sns.barplot(x = 'Year', y = 'Trades', data = trade_data) 
plt.title('Total Trades 2017 vs. 2018')
plt.xlabel('Year')
plt.ylabel('Total Number of Trades')
plt.show()
