#!/usr/bin/env python
# coding: utf-8

# # Average EHR Spending by City

# In[669]:


## Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy import stats
import seaborn as sns


# In[670]:


## Load the datasets
df_ehr_provider = pd.read_csv('EHR_Incentive_Program_Payments_Providers.csv')
df_ehr_hospital = pd.read_csv('EHR_Incentive_Program_Payments_Hospitals.csv')
df_city_revenue = pd.read_csv('City_Revenues_Per_Capita.csv')
df_city_expenditure = pd.read_csv('City_Expenditures_Per_Capita.csv')


# ## Analysis for EHR per Capita for Individual Healthcare Providers

# In[671]:


## delete the unneeded columns on the provider dataset
df_ehr_provider.drop(df_ehr_provider.columns[0:8],axis=1,inplace = True)
df_ehr_provider.drop(df_ehr_provider.columns[1:4],axis=1,inplace = True)
df_ehr_provider.drop(df_ehr_provider.columns[3:6],axis=1,inplace = True)
df_ehr_provider.drop(['Program_Ye','Payee_NPI','Latitude','Longitude','total_rece'],axis=1,inplace = True)
df_ehr_provider.head()


# In[672]:


## group the practitioner data by city and year, then sum the total payments
df_ehr_provider['Business_C'] = df_ehr_provider['Business_C'].str.lower()
df_ehr_provider = df_ehr_provider.groupby(['Business_C','Payment_Ye'])['total_paym'].sum().reset_index()
df_ehr_provider.to_csv('test_group_by_check.csv')
df_ehr_provider.head()


# In[673]:


#change all Entity Names to lower case
df_city_revenue['Entity Name'] = df_city_revenue['Entity Name'].str.lower()
df_city_revenue.head()


# In[674]:


df_city_expenditure['Entity Name'] = df_city_revenue['Entity Name'].str.lower()
df_city_expenditure.head()


# In[675]:


## merge the provider dataset with the city revenue dataset
df_provider_city = df_ehr_provider.merge(df_city_revenue, how="inner",left_on=['Business_C','Payment_Ye'],right_on=['Entity Name','Fiscal Year'])
df_provider_city.head()


# In[676]:


## drop the duplicate columns
df_provider_city = df_provider_city.drop(['Entity Name','Fiscal Year'], axis=1)
df_provider_city.head()


# In[677]:


## merge with the city expenditure dataset
df_provider_city = df_provider_city.merge(df_city_expenditure, how="inner",left_on=['Business_C','Payment_Ye', 'Estimated Population'],right_on=['Entity Name','Fiscal Year', 'Estimated Population'])
df_provider_city = df_provider_city.drop(['Entity Name','Fiscal Year'], axis=1)
df_provider_city.head()


# In[678]:


df_provider_city.to_csv('ultimate_test.csv')


# In[679]:


df_provider_city.head()


# In[680]:


## create the "EHR per Capita" column by divide the total_paym by the estimated population in that year
df_provider_city['EHR Per Capita'] = df_provider_city['total_paym']/df_provider_city['Estimated Population']
df_provider_city.head()


# In[681]:


## Get the last entry for each city as the total payment they have received
df_provider_city_last = df_provider_city.groupby(['Business_C']).apply(lambda x: x.iloc[[-1]]).reset_index(drop=True)
df_provider_city_last.head()


# In[682]:


df_provider_city_last.shape


# In[586]:


df_provider_city_last.to_csv('df_provider_city_last.csv')


# # Outlier treatment: 
# 1. log transformation for 'Expenditures Per Capita'and 'Revenues Per Capita'
# 2. remove outliers using Quantile-based Flooring and Capping
# Quantile-based Flooring and Capping In this technique, we will do the flooring (e.g., the 10th percentile) for the lower values and capping (e.g., the 90th percentile) for the higher values. The lines of code below print the 10th and 90th percentiles of the variable 'Income', respectively. These values will be used for quantile-based flooring and capping.

# In[683]:


import math

print(f"Before treatment: {df_provider_city_last['Expenditures Per Capita'].skew()}")
df_provider_city_last['Expenditures Per Capita'] = df_provider_city_last['Expenditures Per Capita'].apply(lambda x: math.log(x)) # shrink the difference among data
print(f"After treatment: {df_provider_city_last['Expenditures Per Capita'].skew()}")


# In[684]:


print(f"Before treatment: {df_provider_city_last['Revenues Per Capita'].skew()}")
df_provider_city_last['Revenues Per Capita'] = df_provider_city_last['Revenues Per Capita'].apply(lambda x: math.log(x)) # shrink the difference among data
print(f"After treatment: {df_provider_city_last['Revenues Per Capita'].skew()}")


# In[685]:


#determine the upper boundary and lower boundary by plotting box plot
# remove outliers from Seaborn boxplots.

import seaborn as sns 
plt.subplot(1,2,1)
sns.boxplot(x=df_provider_city_last['Revenues Per Capita'])
plt.subplot(1,2,2)
sns.boxplot(x=df_provider_city_last['Expenditures Per Capita'])


# In[686]:


# Boxplot without outliers
plt.subplot(1,2,1)
sns.boxplot(x=df_provider_city_last['Revenues Per Capita'], showfliers = False)
plt.subplot(1,2,2)
sns.boxplot(x=df_provider_city_last['Expenditures Per Capita'], showfliers = False)


# In[687]:


# remove outliers
Q1 = df_provider_city_last['Revenues Per Capita'].quantile(0.25)
Q3 = df_provider_city_last['Revenues Per Capita'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

filter = (df_provider_city_last['Revenues Per Capita'] >= Q1 - 1.5 * IQR) & (df_provider_city_last['Revenues Per Capita'] <= Q3 + 1.5 *IQR)

df_provider_city_last = df_provider_city_last.loc[filter]  

Q1 = df_provider_city_last['Expenditures Per Capita'].quantile(0.25)
Q3 = df_provider_city_last['Expenditures Per Capita'].quantile(0.75)
IQR = Q3 - Q1    #IQR is interquartile range. 

filter = (df_provider_city_last['Expenditures Per Capita'] >= Q1 - 1.5 * IQR) & (df_provider_city_last['Expenditures Per Capita'] <= Q3 + 1.5 *IQR)
df_provider_city_last = df_provider_city_last.loc[filter] 

df_provider_city_last.shape


# ### Plot the preliminary histograms

# In[592]:


## plot the histogram for revenue per capita
plt.hist(df_provider_city_last['Revenues Per Capita'], bins=5000)


# In[593]:


## plot the histogram for expenditure per capita
plt.hist(df_provider_city_last['Expenditures Per Capita'], bins=5000)


# In[594]:


## plot the histogram for EHR per Capita
plt.hist(df_provider_city_last['EHR Per Capita'], bins=5000)


# In[595]:


import sweetviz as sv

my_report = sv.analyze(df_provider_city_last)
my_report.show_html() # Default arguments will generate to "SWEETVIZ_REPORT.html"


# In[688]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# ## Regression Analysis

# In[694]:


# Plot the Distribution plots for the features
import warnings
warnings.filterwarnings('ignore')
plt.subplot(1,3,1)
sns.distplot(df_provider_city_last['Revenues Per Capita'])
plt.subplot(1,3,2)
sns.distplot(df_provider_city_last['Expenditures Per Capita'])
plt.subplot(1,3,3)
sns.distplot(df_provider_city_last['EHR Per Capita']) # right skewed distribution
# ================================================
# Build the model
# ================================================

# Training data
X = df_provider_city_last[['Revenues Per Capita', 'Expenditures Per Capita']] # feature 
y = df_provider_city_last['EHR Per Capita'] # target

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)

# Train the model: include an intercept to the model by default
model = LinearRegression()
model.fit(X_train, y_train)


# In[695]:


y_pred = model.predict(X_test)

# Comparing the test values and the predicted values
comparison_df = pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
comparison_df.head()


# In[696]:


import hvplot.pandas
pd.DataFrame({'True Values(y test)': y_test, 'Predicted Values': y_pred}).hvplot.scatter(x='True Values(y test)', y='Predicted Values')


# In[697]:


# coefficient
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
coeff_df


# Interpreting the coefficients:
# 
# Holding all other features fixed, a 1 unit increase in Revenues Per Capita and Expenditures Per Capita is associated with an increase of 11.678199 and a decrease of -7.991750 in EHR per capita respectively.

# ### Check the distribution of the error terms

# In[698]:


# Residual Histogram
pd.DataFrame({'Error Values': (y_test - y_pred)}).hvplot.kde()


# Check the distribution of the error terms
# In linear regression we assume that the error term follows normal distribution. So we have to check this assumption before we can use the model for making predictions. We check this by looking at the histogram of the error term visually, making sure that the error terms are normally distributed around zero and that the left and right side are broadly similar.

# ### Evaluate the regression

# In[656]:


from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def cross_val(model):
    pred = cross_val_score(model, X, y, cv=10)
    return pred.mean()

def print_evaluate(true, predicted):  
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square


# In[699]:


test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


# In[700]:


#  don't include an intercept in your model:
model = LinearRegression(fit_intercept = False)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


# # Comparing machine learning models

# ## Application of Decision Tree regression

# In[658]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

model = DecisionTreeRegressor(random_state = 0)
model.fit(X_train, y_train)
mae=metrics.mean_absolute_error(y_test, y_pred)
mse=metrics.mean_squared_error(y_test, y_pred)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Decision Tree regression", *evaluate(y_test, test_pred) , cross_val(DecisionTreeRegressor())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# ## Application of Random Forest Regression

# In[692]:


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators = 300 ,  random_state = 0)
model.fit(X_train,y_train)
#Predicting the SalePrices using test set 
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Random Forest Regression", *evaluate(y_test, test_pred) , cross_val(RandomForestRegressor())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# ## Application of Support Vector Regression

# In[660]:


from sklearn.svm import SVR
model= SVR(kernel='rbf')
model.fit(X_train,y_train)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Support Vector Regression", *evaluate(y_test, test_pred) , cross_val(SVR())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# ## Random Sample Consensus(RANSAC) Regression
# Random sample consensus (RANSAC) is an iterative method to estimate parameters of a mathematical model from a set of observed data that contains outliers, when outliers are to be accorded no influence on the values of the estimates. Therefore, it also can be interpreted as an outlier detection method.
# 
# A basic assumption is that the data consists of "inliers", i.e., data whose distribution can be explained by some set of model parameters, though may be subject to noise, and "outliers" which are data that do not fit the model. The outliers can come, for example, from extreme values of the noise or from erroneous measurements or incorrect hypotheses about the interpretation of data. RANSAC also assumes that, given a (usually small) set of inliers, there exists a procedure which can estimate the parameters of a model that optimally explains or fits this data.

# In[607]:


from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold

model = RANSACRegressor(base_estimator=LinearRegression(), max_trials=100)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Random Sample Consensus", *evaluate(y_test, test_pred) , cross_val(RANSACRegressor())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# ## Stochastic Gradient Descent
# Gradient Descent is a very generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Sescent is to tweak parameters iteratively in order to minimize a cost function. Gradient Descent measures the local gradient of the error function with regards to the parameters vector, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum.

# In[661]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_reg.fit(X_train, y_train)

test_pred = sgd_reg.predict(X_test)
train_pred = sgd_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred),cross_val(SGDRegressor())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# ## Models Comparison

# In[609]:


# MAE
results_df.set_index('Model', inplace=True)
results_df['MAE'].plot(kind='barh', figsize=(12, 8))


# In[611]:


#MSE
results_df['MSE'].plot(kind='barh', figsize=(12, 8))


# In[612]:


#RMSE
results_df['RMSE'].plot(kind='barh', figsize=(12, 8))


# In[613]:


#R2 Square
results_df['R2 Square'].plot(kind='barh', figsize=(12, 8))


# # Kruskal-Wallis Test
# 
# Assumptions:
# The variables should have:
# 1. One independent variable with two or more levels (independent groups) -> city (independent variable)
# 2. Ordinal scale, Ratio Scale or Interval scale dependent variables. -> 'EHR Per Capita' is a ratio variable.
# 
# We want to find out how city affects actual EHR per capita.
# 
# Null Hypothesis: the EHR per capita has the same distributions in each city

# In[626]:


# Importing required library
from scipy.stats import kruskal

# Perform Kruskal Wallis Test
group_data = df_provider_city_last.groupby('Business_C')['EHR Per Capita'].apply(list).to_dict()

result = stats.kruskal(*group_data.values())
result


# In[628]:


#Interpretation of the results at 5% level of significance
# Level of significance
alpha = 0.05
if result.pvalue < alpha:
    print('Reject Null Hypothesis (Different distribution')
else:
    print('Do not Reject Null Hypothesis (Same distribution)')


# # Spearman's rank correlation coefficient
# 
# H0: EHR per capita is not related to the Revenues Per Capita.
# 
# H1: EHR per capita is related to the Revenues Per Capita (Higher per-capita revenue city gets more or less per-capita EHR). 

# In[707]:


from scipy import stats
result = stats.spearmanr(df_provider_city_last['EHR Per Capita'],df_provider_city_last['Revenues Per Capita'] )
print(result)
#Interpretation of the results at 5% level of significance
# Level of significance
alpha = 0.05
if result.pvalue < alpha:
    print('Reject Null Hypothesis')
else:
    print('Do not Reject Null Hypothesis')


# H0: EHR per capita is not related to the Expenditures Per Capita.
#     
# H1: EHR per capita is related to the Expenditures Per Capita. 

# In[708]:


result = stats.spearmanr(df_provider_city_last['EHR Per Capita'],df_provider_city_last['Expenditures Per Capita'] )
print(result)
#Interpretation of the results at 5% level of significance
# Level of significance
alpha = 0.05
if result.pvalue < alpha:
    print('Reject Null Hypothesis')
else:
    print('Do not Reject Null Hypothesis')


# In[710]:


from scipy.stats import friedmanchisquare

# compare samples
stat, p = friedmanchisquare(df_provider_city_last['EHR Per Capita'], df_provider_city_last['Expenditures Per Capita'], df_provider_city_last['Revenues Per Capita'])
print('Statistics=%.3f, p=%.3f' % (stat, p))
# interpret
alpha = 0.05
if p > alpha:
	print('Same distributions (fail to reject H0)')
else:
	print('Different distributions (reject H0)')


# # Reduce variable to make the R2 square larger

# In[634]:


# Plot the Distribution plots for the features
import warnings
warnings.filterwarnings('ignore')
plt.subplot(1,2,1)
sns.distplot(df_provider_city_last['Expenditures Per Capita'])
plt.subplot(1,2,2)
sns.distplot(df_provider_city_last['EHR Per Capita']) # right skewed distribution
# ================================================
# Build the model
# ================================================

# Training data
X = df_provider_city_last[['Expenditures Per Capita']] #  feature matrix
y = df_provider_city_last['EHR Per Capita'] # target matrix

from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7,test_size=0.3,random_state=100)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# In[635]:


test_pred = model.predict(X_test)
train_pred = model.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df = pd.DataFrame(data=[["Linear Regression", *evaluate(y_test, test_pred) , cross_val(LinearRegression())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])


# In[636]:


from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

model = DecisionTreeRegressor(random_state = 0)
model.fit(X_train, y_train)
mae=metrics.mean_absolute_error(y_test, y_pred)
mse=metrics.mean_squared_error(y_test, y_pred)
test_pred = model.predict(X_test)
train_pred = model.predict(X_train)
print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Decision Tree regression", *evaluate(y_test, test_pred) , cross_val(DecisionTreeRegressor())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[637]:


from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(n_iter_no_change=250, penalty=None, eta0=0.0001, max_iter=100000)
sgd_reg.fit(X_train, y_train)

test_pred = sgd_reg.predict(X_test)
train_pred = sgd_reg.predict(X_train)

print('Test set evaluation:\n_____________________________________')
print_evaluate(y_test, test_pred)

results_df_2 = pd.DataFrame(data=[["Stochastic Gradient Descent", *evaluate(y_test, test_pred),cross_val(SGDRegressor())]], 
                          columns=['Model', 'MAE', 'MSE', 'RMSE', 'R2 Square', "Cross Validation"])
results_df = results_df.append(results_df_2, ignore_index=True)


# In[ ]:




