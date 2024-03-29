# -*- coding: utf-8 -*-
"""
Eric Born
Utilizing various ML algorithms to predict stock prices
"""

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score
#from sklearn.model_selection import train_test_split

'''
Takes x and y as inputs and calculates the size of the array,
the sum of squares, slope and intercept. 
Used in the Linear Regression section
'''
def estimate_coef (x, y):
    n = np.size(x)
    mu_x, mu_y = np.mean(x), np.mean(y)
    SS_xy = np.sum(y * x) - n * mu_y * mu_x
    SS_xx = np.sum(x * x) - n * mu_x * mu_x
    slope = SS_xy / SS_xx
    intercept = mu_y - slope * mu_x
    return (slope, intercept)

# Using the stock ticker BSX - Boston Scientific Corporation.
# setup input directory and filename
ticker = 'BSX-labeled'
input_dir = r'C:\Users\Eric\Documents\GitHub\stock-prediction\data'
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
    bsx_df = pd.read_csv(ticker_file)
    print('opened file for ticker: ', ticker,'\n')

except Exception as e:
    print(e)
    sys.exit('failed to read stock data for ticker: ', ticker)

# Create class column where red = 0 and green = 1
bsx_df['class'] = bsx_df['label'].apply(lambda x: 1 if x =='green' else 0)

# Create separate dataframes for 2017 and 2018 data
# 2017 will be used as training, 2018 as testing for the model
bsx_df_2017 = bsx_df.loc[bsx_df['td_year']==2017]
bsx_df_2018 = bsx_df.loc[bsx_df['td_year']==2018]

# Reset indexes
bsx_df_2017 = bsx_df_2017.reset_index(level=0, drop=True)
bsx_df_2018 = bsx_df_2018.reset_index(level=0, drop=True)

# Create reduced dataframe only containing week number, mu, sig and label
bsx_2017_reduced = pd.DataFrame( {'week nbr' : range(1, 53),
                'mu'    : bsx_df_2017.groupby('td_week_number')['return'].mean(),
                'sig'   : bsx_df_2017.groupby('td_week_number')['return'].std(),
                'label' : bsx_df_2017.groupby('td_week_number')['class'].first()})

# Create reduced dataframe only containing week number, mu, sig and label
bsx_2018_reduced = pd.DataFrame( {'week nbr' : range(0, 53),
                'mu'    : bsx_df_2018.groupby('td_week_number')['return'].mean(),
                'sig'   : bsx_df_2018.groupby('td_week_number')['return'].std(),
                'label' : bsx_df_2018.groupby('td_week_number')['class'].first()})

# Replacing nan in week 52 sigma column with a zero due to 
# there being only 1 trading day that week.
bsx_2018_reduced = bsx_2018_reduced.fillna(0)

# remove index name labels from dataframes
del bsx_2017_reduced.index.name
del bsx_2018_reduced.index.name

# Define features labels
features = ['mu', 'sig']

#####
# train/test and scale data
#####

# Initialize scaler
scaler = StandardScaler()

# Create x training and test sets from 2017/2018 features values.
x_train = bsx_2017_reduced[features].values
x_test = bsx_2018_reduced[features].values

# create y training and test sets from 2017/2018 label values
y_train = bsx_2017_reduced['label'].values
y_test = bsx_2018_reduced['label'].values

# Scaler for training data
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)

# Scaler for test data
scaler.fit(x_test)
x_test_scaled  = scaler.transform(x_test)

# stores adj_close values for the last day of each trading week
adj_close = bsx_df_2018.groupby('td_week_number')['adj_close'].last()

# stores open price for the first day of each trading week
open_price = bsx_df_2018.groupby('td_week_number')['open'].first()

#####
# End setup
#####

#####
# Start KNN
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
        knn_classifier.fit(x_train_scaled, y_train)

        # Perform predictions
        pred_k = knn_classifier.predict(x_test_scaled)

        # Store error rate and accuracy for particular K value
        k_value.append(k)
        error_rate.append(round(np.mean(pred_k != y_test) * 100, 2))
        accuracy.append(round(sum(pred_k == y_test) / len(pred_k) * 100, 2))

except Exception as e:
    print(e)
    sys.exit('failed to build the KNN classifier.')

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

# Create the classifier with neighbors set to 7
knn_2018 = KNeighborsClassifier(n_neighbors = 4)

# Train the classifier using all of 2017 data
knn_2018.fit(x_train_scaled, y_train)
        
# Perform predictions on 2018 data
knn_pred_2018 = knn_2018.predict(x_test_scaled)

# Capture error and accuracy rates for 2018 predictions
knn_error_2018 = round(np.mean(knn_pred_2018 != y_test) * 100, 2)
knn_accuracy_2018 = round(sum(knn_pred_2018 == y_test) / 
                          len(knn_pred_2018) * 100, 2)

# print accuracy
print('\nThe accuracy on 2018 data when K = 4 is:', knn_accuracy_2018, '%')

# Output the confusion matrix
cm = confusion_matrix(y_test, knn_pred_2018)
tn, fp, fn, tp  = confusion_matrix(y_test, knn_pred_2018).ravel()
print('\nConfusion matrix for KNN:')
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
plt.title('KNN Confusion Matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(y_test, knn_pred_2018) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

# Implement a trading strategy based upon label predicitons 
# Initialize wallet and shares to track current money and number of shares.
knn_wallet = 100.00
knn_shares = 0
knn_profit = 0
knn_worth = 0

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for i in range(0, len(knn_pred_2018)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if knn_pred_2018[i] == 0 and knn_shares > 0:
            knn_wallet = round(knn_shares * adj_close[i - 1], 2)
            shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if knn_pred_2018[i] == 1 and knn_shares == 0: 
            knn_shares = knn_wallet / open_price[i]
            knn_wallet = 0            
            
except Exception as e:
    print(e)
    sys.exit('Failed to evaluate df_2018 labels')

# set worth by multiplying stock price on final day by total shares
knn_worth = round(shares * adj_close[52], 2)

if knn_worth == 0:
    knn_worth = knn_wallet
    knn_profit = round(knn_wallet - 100.00, 2)
else:
    knn_profit = round(knn_worth - 100.00, 2)

#####
# Logistic Regression
#####

# Create a logistic classifier
# set solver to avoid the warning
log_reg_classifier = LogisticRegression(solver = 'liblinear')

# Train the classifier on 2017 data
log_reg_classifier.fit(x_train_scaled, y_train)

# Predict using 2018 feature data
log_reg_prediction = log_reg_classifier.predict(x_test_scaled)

# Print the mu and sig
print('output the coefficients with feature names')
coef = log_reg_classifier.coef_
for p,c in zip(features,list(coef[0])):
    print(p + '\t' + str(c))

# 79.25% accuracy for year 2
accuracy = np.mean(log_reg_prediction == y_test)
print('\nThe accuracy for year 2 is:')
print(round(accuracy * 100, 2), '%')

# Output the confusion matrix
cm = confusion_matrix(y_test, log_reg_prediction)
tn, fp, fn, tp  = confusion_matrix(y_test, log_reg_prediction).ravel()
print('\nConfusion matrix for Logistic Regression:')
print(cm, '\n')

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="BrBG" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Log Reg Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# true positive rate (sensitivity or recall) and true negative rate 
# (specificity) for year 2
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(y_test, log_reg_prediction) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

# Log reg trading strategy based upon label predicitons
# Initialize wallet and shares to track current money and number of shares.
log_reg_wallet = 100.00
log_reg_shares = 0
log_reg_worth = 0

# stores adj_close values for the last day of each trading week
#adj_close = bsx_df_2018.groupby('td_week_number')['adj_close'].last()
#
## stores open price for the first day of each trading week
#open_price = bsx_df_2018.groupby('td_week_number')['open'].first()

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for i in range(0, len(log_reg_prediction)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if log_reg_prediction[i] == 0 and log_reg_shares > 0:
            log_reg_wallet = round(log_reg_shares * adj_close[i - 1], 2)
            log_reg_shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if log_reg_prediction[i] == 1 and log_reg_shares == 0: 
            log_reg_shares = log_reg_wallet / open_price[i]
            log_reg_wallet = 0            
            
except Exception as e:
    print(e)
    print('Failed to evaluate df_2018 labels')


# set worth by multiplying stock price on final day by total shares
log_reg_worth = round(log_reg_shares * adj_close[52], 2)

if log_reg_worth == 0:
    log_reg_worth = log_reg_wallet
    log_reg_profit = round(log_reg_wallet - 100.00, 2)
else:
    log_reg_profit = round(log_reg_worth - 100.00, 2)

######
# Linear Regression
######

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

# Creates a linear regression object
lm = LinearRegression()

# stores the position 0, 1, -1 for each window size
position_2017_df  = pd.DataFrame()

# list that will be populated from the below for loop
# to contains the window sizes
window_size = []

# For loop iterates through bsx_df_2017 with a window size of 5 to 30, 
# incrementing by 1. The window size is used as the number of days in the batch
# being evaluated at a time.
try:
    for window in range(5,31):
        # Create position column to indicate whether its a buy or sell day.
        # column is reset to all 0's at the start of each loop iteration
        bsx_df_2017['position'] = 0
        bsx_df_2017['prediction'] = 0
        
        # window size list populated with size increments
        window_size.append(window)
        
        # set window_end equal to window - 1 due to zero index
        window_start = 0
        window_end = window - 1
        
        # loop that handles gathering the adj_close and close price 
        # for the appropriate window size
        for rows in range(0, len(bsx_df_2017)):
            adj_close = np.array(bsx_df_2017.loc[window_start:window_end,
                                     'adj_close']).reshape(-1, 1)
            close = np.array(bsx_df_2017.loc[window_start:window_end,
                                     'close']).reshape(-1, 1)
            lm.fit(adj_close, close)
            
            # Breaks on the last row since it cannot predict w + 1 if 
            # there is no data for the next day, else it creates
            # a prediction.
            if window_end == len(bsx_df_2017) - 1:
                break
            else:
                pred = lm.predict(np.array(bsx_df_2017.loc[window_end + 1, 
                                          'adj_close']).reshape(-1, 1))
            
            # store the predicted value in the 2017 dataframe
            bsx_df_2017.loc[window_end + 1, 'prediction'] = float(pred) 
    
            # updates the position column with a 1 when prediciton for 
            # tomorrows close price (w + 1) is greater than the close price of 
            # w. Else it marks it with a -1 to indicate a lower price.
            if float(pred) > bsx_df_2017.loc[window_end, 'close']:
                bsx_df_2017.loc[window_end, 'position'] = 1
            elif float(pred) == bsx_df_2017.loc[window_end, 'close']:
                bsx_df_2017.loc[window_end, 'position'] = 0
            else:
                bsx_df_2017.loc[window_end, 'position'] = -1
            window_start += 1
            window_end += 1
        
        # writes the position column to a the position dataframe after each
        # window iteration
        position_2017_df[str(window)] = bsx_df_2017.loc[:, 'position']

except Exception as e:
    print(e)
    sys.exit('Failed to perform predictions on 2017 data')
    
# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
lin_reg_2017_shares = 0
lin_reg_2017_worth = 0
lin_reg_2017_price = 0
lin_reg_2017_name_increment = 5
trade_data_2017_df = pd.DataFrame()

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_2017_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        lin_2017_price_name  = 'lin_2017_price'  + \
                                str(lin_reg_2017_name_increment)
        lin_2017_shares_name = 'lin_2017_shares' + \
                                str(lin_reg_2017_name_increment)
        lin_2017_worth_name  = 'lin_2017_window '  + \
                                str(lin_reg_2017_name_increment)

        for position_row in range(0, len(position_2017_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_2017_df.iloc[position_row, position_column] == 1 
            and lin_reg_2017_shares == 0): 
                lin_reg_2017_shares = 100.00 / bsx_df_2017.loc[position_row,
                                                               'close']           
                lin_reg_2017_long_price = bsx_df_2017.loc[position_row,
                                                          'close']
                trade_data_2017_df.at[position_row, lin_2017_price_name] = \
                lin_reg_2017_long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_2017_df.iloc[position_row, position_column] == -1
            and lin_reg_2017_shares != 0): 
                long_worth = ((lin_reg_2017_shares 
                              * bsx_df_2017.loc[position_row, 'close'])
                              - lin_reg_2017_long_price * lin_reg_2017_shares)
                trade_data_2017_df.at[position_row, lin_2017_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_2017_df.at[position_row, lin_2017_worth_name] = (
                                            bsx_df_2017.loc[position_row, 
                                                            'close'])
                lin_reg_2017_shares = 0
                lin_reg_2017_price = 0
                lin_reg_2017_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_2017_df.at[position_row, lin_2017_shares_name] = \
            lin_reg_2017_shares

        # increments the name_increment to represent the window size
        lin_reg_2017_name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('Failed to build trading data for trade_data_df')            

# creates a list containing the column names from the trade_data_df
name_list = []
for column in range(2, len(trade_data_2017_df.iloc[0, :]), 3):
    name_list.append(trade_data_2017_df.iloc[:, column].name)

# create a dataframe to store the daily profits made from selling stocks
summary_2017_df = trade_data_2017_df[name_list].copy()

# Sum of profits by window size
profit_2017 = summary_2017_df.sum()
print(profit_2017)

# highest 2017 profit
profit_2017_best = round(profit_2017[0], 2)

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

# best mean
mean_2017_best = round(mean_2017[-1],2)

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

# Above it was determined that using a fixed window size of 5 is the most
# profitable, since there are more overall trades and the average per trade
# is only 0.35 or less per trade. This window size will be used to analyze and 
# predict stock prices in the 2018 data.
# Unfortunately the two measures being used, closing and adjusted 
# closing price are exactly the same every day in 2018
# so the results are a bit uninteresting

# Setup column in the 2018 dataframe to track the buy/sell position,
# the predicted value and initial start and end window positions
bsx_df_2018['position'] = 0
bsx_df_2018['prediction'] = 0
window_start = 0
window_end = 4
fixed_window = 5

# stores the position 0, 1, -1 for each window size
position_2018_df  = pd.DataFrame()
try:    
    # loop that handles gathering the adj_close and close price 
    # for the 2018 dataframe
    for rows in range(0, len(bsx_df_2018)):
        adj_close = np.array(bsx_df_2018.loc[window_start:window_end,
                                 'adj_close']).reshape(-1, 1)
        close = np.array(bsx_df_2018.loc[window_start:window_end,
                                 'close']).reshape(-1, 1)
        # fits the model using adjusted close and close stock prices
        lm.fit(adj_close, close)
        
        # Breaks on the last row since it cannot predict w + 1 if 
        # there is no data for the next day, else it creates
        # a prediction.
        if window_end == len(bsx_df_2018) - 1:
            break
        else:
            pred = lm.predict(np.array(bsx_df_2018.loc[window_end + 1, 
                                      'close']).reshape(-1, 1))
        
        # store the predicted value in the 2018 dataframe
        bsx_df_2018.loc[window_end + 1, 'prediction'] = float(pred) 
        
        # updates the position column with a 1 when prediciton for tomorrows
        # close price (w + 1) is greater than the close price of w.
        # Else it marks it with a -1 to indicate a lower price.
        if float(pred) > bsx_df_2018.loc[window_end, 'close']:
            bsx_df_2018.loc[window_end, 'position'] = 1
        elif float(pred) == bsx_df_2018.loc[window_end, 'close']:
            bsx_df_2018.loc[window_end, 'position'] = 0
        else:
            bsx_df_2018.loc[window_end, 'position'] = -1
        window_start += 1
        window_end += 1

        # writes the position column to a the position dataframe after each
        # window iteration
        position_2018_df[str(fixed_window)] = bsx_df_2018.loc[:, 'position']

except Exception as e:
    print(e)
    sys.exit('Failed to build prediction data for 2018')

# Initialize variables and trade_data_df to their starting values
# before they are utilized in the loop to build out the trade_data df
lin_reg_2018_shares = 0
lin_reg_2018_worth = 0
lin_reg_2018_price = 0
lin_reg_2018_name_increment = 5
trade_data_2018_df = pd.DataFrame()

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for position_column in range(0, len(position_2018_df.iloc[0, :])):
        # used to increment the column name to represend the window size
        lin_2018_price_name  = 'lin_2018_price'  + \
                                str(lin_reg_2018_name_increment)
        lin_2018_shares_name = 'lin_2018_shares' + \
                                str(lin_reg_2018_name_increment)
        lin_2018_worth_name  = 'lin_2018_window '  + \
                                str(lin_reg_2018_name_increment)

        for position_row in range(0, len(position_2018_df)):
            # Buy section
            # long_shares buy should occur if position dataframe  
            # contains a 1 and there are no long_shares held
            if (position_2018_df.iloc[position_row, position_column] == 1 
            and lin_reg_2018_shares == 0): 
                lin_reg_2018_shares = 100.00 / bsx_df_2018.loc[position_row,
                                                               'close']           
                lin_reg_2018_long_price = bsx_df_2018.loc[position_row,
                                                          'close']
                trade_data_2018_df.at[position_row, lin_2018_price_name] = \
                lin_reg_2018_long_price
                #trade_data_df.at[position_row, long_worth_name] = ((long_shares 
                #              * df_2017.loc[position_row, 'close'])
                #              - long_price * long_shares)
            
            # Sell section
            # long_shares sell should occur if position dataframe  
            # contains a -1 and there are long_shares held
            if (position_2018_df.iloc[position_row, position_column] == -1
            and lin_reg_2018_shares != 0): 
                long_worth = ((lin_reg_2018_shares 
                              * bsx_df_2018.loc[position_row, 'close'])
                              - lin_reg_2018_long_price * lin_reg_2018_shares)
                trade_data_2018_df.at[position_row, lin_2018_worth_name] = (
                                                          round(long_worth, 2))
                trade_data_2018_df.at[position_row, lin_2018_worth_name] = (
                                            bsx_df_2018.loc[position_row, 
                                                            'close'])
                lin_reg_2018_shares = 0
                lin_reg_2018_price = 0
                lin_reg_2018_worth = 0
                  
            # On each loop iteration record the current long shares held
            trade_data_2018_df.at[position_row, lin_2018_shares_name] = \
            lin_reg_2018_shares

        # increments the name_increment to represent the window size
        lin_reg_2018_name_increment += 1
            
except Exception as e:
    print(e)
    sys.exit('Failed to build trading data for trade_data_df')            

# Finish data building
    
#########
    
# Start analysis/presenation
    
# create a dataframe to store the daily profits made from selling stocks
summary_2018_df = trade_data_2018_df['lin_2018_window 5'].copy()

# Creating an estimated coefficient between the measures
# the slope is a perfect 1 and the intercept is at 0
# dataframes start at position 5 since the first 0-4 were not predicted
# do to window size starting at 5
coefficient = estimate_coef(bsx_df_2018.loc[5:,'adj_close'],
                            bsx_df_2018.loc[5:,'prediction'])
print(coefficient)

# Generate a plot of the actual vs predicted values
sns.scatterplot(bsx_df_2018.loc[5:,'adj_close'],
                bsx_df_2018.loc[5:,'prediction'], color='navy')
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
long_days = bsx_df_2018[bsx_df_2018['position'] > 0].count()['position']
print(long_days)

# short days
# 111
short_days = bsx_df_2018[bsx_df_2018['position'] < 0].count()['position']
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
profit_2018 = round(summary_2018_df.sum(), 2)
print(profit_2018)

# 2017 vs 2018 profits
print('Profits 2017:', profit_2017_best,'\nProfits 2018:', profit_2018)

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

# 2017 vs 2018 means
print('Mean 2017:', mean_2017_best,'\nMean 2018:', mean_2018)

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

# 2017 vs 2018 total trades
print('2017 max trades:', trades_2017[0],'\n2018 trades:', trades_2018)

# Plot total number of trades 2017 vs 2018
trade_data = pd.DataFrame({'Year': ['2017', '2018'], 
                            'Trades': [int(trades_2017[0]), int(trades_2018)]})
sns.barplot(x = 'Year', y = 'Trades', data = trade_data) 
plt.title('Max Trades 2017 vs. 2018')
plt.xlabel('Year')
plt.ylabel('Max Number of Trades')
plt.show()

#####
# Decision Tree
#####

# Create a decisions tree classifier
tree_clf = tree.DecisionTreeClassifier(criterion = 'entropy')

# Train the classifier on 2017 data
tree_clf = tree_clf.fit(x_train, y_train)

# Predict using 2018 feature data
tree_prediction = tree_clf.predict(x_test)

# calculate error rate
tree_accuracy_rate = 100-(round(np.mean(tree_prediction != y_test) * 100, 2))

# Print error rate
print('The decision tree classifier has an accuracy of', 
      tree_accuracy_rate,'%')

# Output the confusion matrix
cm = confusion_matrix(y_test, tree_prediction)
print('\nDecision Tree Confusion matrix:')
print(cm, '\n')

# Create confusion matrix heatmap
# setup class names and tick marks
class_names=[0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Create heatmap and labels
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="summer" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Decision Tree Confusion Matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

# store confusion matrix figures
tn, fp, fn, tp = confusion_matrix(y_test, tree_prediction).ravel()

# TPR/TNR rates
print('The TPR is:', str(tp) + '/' + str(tp + fn) + ',',
      round(recall_score(y_test, tree_prediction) * 100, 2),'%')
print('The TNR is:', str(tn) + '/' + str(tn + fp) + ',',
    round(tn / (tn + fp) * 100, 2),'%')

# Trading strategy based upon label predicitons 
# Initialize wallet and shares to track current money and number of shares.
tree_wallet = 100.00
tree_shares = 0
tree_worth = 0
tree_profit = 0

# for loop that evaluates the dataset deciding when to buy/sell based
# upon the prediction labels. 0 is a bad week, 1 is a good week
try:
    for day in range(0, len(tree_prediction)):
        # Sell should occur on the last day of a green week at 
        # the adjusted_close price. Since i is tracking the current
        # trading week we need to minus 1 to get the adjusted close price
        # from the previous trading week
        if tree_prediction[day] == 0 and tree_shares > 0:
            tree_wallet = round(tree_shares * adj_close[-1][0], 2)
            tree_shares = 0
            
        # Buy should occur on the first day of a green week at the open price
        if tree_prediction[day] == 1 and tree_shares == 0: 
            tree_shares = tree_wallet / open_price[day]
            tree_wallet = 0            
            
except Exception as e:
    print(e)
    exit('Failed to evaluate decision tree labels')

# set worth by multiplying stock price on final day by total shares
tree_worth = round(tree_shares * adj_close[-1][0], 2)

if tree_worth == 0:
    tree_worth = tree_wallet
    tree_profit = round(tree_wallet - 100.00, 2)
else:
    tree_profit = round(tree_worth - 100.00, 2)

#####
# End decision tree
#####

#####
# Setup for buy and hold strat
#####

# Buy and hold
# Initialize wallet and shares to track current money and number of shares.
bh_wallet = 100.00
bh_shares = 0
bh_profit = 0
bh_worth = 0

# Calculate shares, worth and profit
shares = round(bh_wallet / float(open_price[0]), 6)
bh_worth = round(shares * adj_close[-1][0], 2)
bh_profit = round(bh_worth - 100.00, 2)

#####
# Output profits from each algorithm
#####

# KNN profits
print('\nKNN Label Strategy:')
print('Total Cash: $', knn_wallet, '\nTotal shares:', round(knn_shares, 6),
      '\nWorth: $', knn_worth)    
print('This method would close the year at $', knn_worth, 'a profit of $', 
      knn_profit)

# Logistic regression
print('\nLogistic Regression Label Strategy:')
print('Total Cash: $', log_reg_wallet, '\nTotal shares:',
      round(log_reg_shares, 6), '\nWorth: $', log_reg_worth)    
print('This method would close the year at $', log_reg_worth, 'a profit of $',
      log_reg_profit)

# Decision Tree profits
print('\nDecisions tree Label Strategy:')
print('Total Cash: $', "%.2f"%tree_wallet, '\nTotal shares:',
      round(tree_shares, 6), '\nWorth: $', "%.2f"%tree_worth)
print('This method would close the year at $', "%.2f"%tree_worth,
      'a profit of $', "%.2f"%tree_profit)

# Buy and hold profits
print('\n2018 buy and hold:','\nCurrently own', bh_shares, 'shares',
      '\nWorth','$',"%.2f"%round(bh_worth, 2))
print('Selling on the final day would result in $',"%.2f"%bh_worth, 
      'a profit of $', "%.2f"%bh_profit)