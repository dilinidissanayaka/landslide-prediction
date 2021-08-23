#!/usr/bin/env python
# coding: utf-8

# #  *Predicting Landslide Vulnarability From Rainfall Dataset*

# ### Import packages

# In[117]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import itertools
from statsmodels.graphics.tsaplots import plot_acf ,plot_pacf

import warnings
warnings.filterwarnings("ignore")

from sklearn import linear_model


label_encoder = LabelEncoder()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### import dataset

# In[118]:


import os
for dirname, _, filenames in os.walk('C:/Users/Dilini Dissanayaka/Desktop/Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename)) 


# In[119]:


landslide=pd.read_excel('C:/Users/Dilini dissanayaka/Desktop/Dataset\landslde data.xlsx')
rainfall=pd.read_excel(r'C:/Users/Dilini dissanayaka/Desktop/Dataset\rainfall.xlsx')


# In[120]:


landslide.head()


# In[38]:


landslide.describe()


# In[39]:


rainfall.head()


# In[40]:



rainfall['station_name'].unique()


# In[41]:


rainfall.groupby('station_name')['yyyy'].count()


# In[42]:


rainfall.describe()


# ### Landslide data
# 
# # Checking missing values

# In[43]:


print(landslide.shape)

missing_values_landslide = (landslide.isnull().sum())
print(missing_values_landslide[missing_values_landslide > 0])


# In[44]:


landslide


# High number of missing values in Report No and Location. So droping both columns.


# In[45]:


print(landslide.shape)

missing_values_landslide = (landslide.isnull().sum())
print(missing_values_landslide[missing_values_landslide > 0])


# In[46]:


landslide


# Droping rows with missing data


# In[47]:


landslide.dropna(axis=0, subset=['year'], inplace=True)
landslide.dropna(axis=0, subset=['month'], inplace=True)


# In[48]:


print(landslide.shape)

missing_values_landslide = (landslide.isnull().sum())
print(missing_values_landslide[missing_values_landslide > 0])


# In[49]:


landslide
# Landslide data is ready.



# ### Rainfall Data

# In[50]:


print(rainfall.shape)

missing_values_landslide = (rainfall.isnull().sum())
print(missing_values_landslide[missing_values_landslide > 0])


# In[51]:


rainfall = rainfall.drop('abbreviation', axis=1)
rainfall = rainfall.drop('code', axis=1)
rainfall = rainfall.drop('elevation', axis=1)
rainfall = rainfall.drop('longitude', axis=1)
rainfall = rainfall.drop('latitude', axis=1)


# In[52]:


rainfall 


# In[53]:


rainfall.dropna(axis=0, subset=['yyyy'], inplace=True)
rainfall.dropna(axis=0, subset=['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC'], inplace=True)
rainfall.dropna(axis=0, subset=['annual'], inplace=True)
#rainfall.dropna(axis=0, subset=['monthly average'], inplace=True)


# In[54]:


rainfall.head()


# In[55]:


rainfall['station_name'].unique()


# Converting categorical data into numerical data


# In[56]:


rainfall['station_name']


# 

# In[57]:


#rainfall['station_name'] = rainfall['station_name'].astype(str)


# In[58]:


#rainfall['station_name'] = label_encoder.fit_transform(rainfall['station_name'])


# In[59]:


rainfall


# In[60]:


#rainfall['station_name']


# 
# ## rainfall data is ready.

# # visualization 
# 

# 
# ### visualizing rainfall

# In[61]:



subdivision = rainfall['station_name'].unique()
subdivision


# In[62]:



plt.style.use('ggplot')

fig = plt.figure(figsize=(16, 8))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='station_name', y='yyyy', data=rainfall)
ax = plt.title('Annual rainfall in all States and UT')

ax = plt.subplot(2,1,2)
ax = plt.xticks(rotation=90)
ax = sns.barplot(x='station_name', y='yyyy', data=rainfall)


# In[63]:



# monthly rainfall


# In[64]:



ax=rainfall.groupby("yyyy").mean()['annual'].plot(ylim=(90,300),color='b',marker='o',linestyle='-',linewidth=2,figsize=(12,8));
rainfall['MA10'] = rainfall.groupby('yyyy').mean()['annual'].rolling(100).mean()
rainfall.MA10.plot(color='r',linewidth=4)

plt.xlabel('Year',fontsize=20)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('Annual Rainfall  from Year 1999 to 2019',fontsize=25)
ax.tick_params(labelsize=15)
plt.grid()
plt.ioff()


# In[65]:


ax=rainfall[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot.bar(width=0.5,edgecolor='k',align='center',linewidth=2,figsize=(16,8))
plt.xlabel('Month',fontsize=30)
plt.ylabel('Monthly Rainfall (in mm)',fontsize=20)
plt.title('Monthly Rainfall ',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# 
# ## heat map of rainfall

# In[66]:


fig=plt.gcf()
fig.set_size_inches(15,15)
fig=sns.heatmap(rainfall.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# In[67]:


ax=rainfall.groupby(['station_name'])['annual'].max().sort_values().tail(20).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
#ax=rainfall.plot.bar(x='station_name',y='annual',width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
plt.xlabel('station',fontsize=30)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('station with Maximum Rainfall ',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# 
# # visualizing landslide

# In[68]:


plt.style.use('ggplot')

fig = plt.figure(figsize=(16, 8))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='station_name', y='yyyy', data=rainfall)
ax = plt.title('Annual rainfall in all States and UT')

ax = plt.subplot(2,1,2)
ax = plt.xticks(rotation=90)
ax = sns.barplot(x='station_name', y='yyyy', data=rainfall)


# In[69]:


print('Average annual rainfall received ]=',int(rainfall['annual'].mean()),'mm')


# # comparison

# ### highest rainfall

# In[70]:


rainfall.groupby("yyyy").mean()['annual'].sort_values(ascending=False).head(10)


# ## highest landslide

# In[122]:


landslide.groupby("year").mean()['annual average'].sort_values(ascending=False).head(10)


# ## lowest rainfall

# In[71]:


rainfall.groupby("yyyy").mean()['annual'].sort_values(ascending=False).tail(10)


# ## lowest landslide

# In[124]:


landslide.groupby("year").mean()['annual average'].sort_values(ascending=False).tail(10)


# In[72]:



rainfall.groupby(['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']).mean()['monthly average'].sort_values(ascending=False).head(10) 


# In[73]:


rainfall.groupby(['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']).mean()['monthly average'].sort_values(ascending=False).tail(10)


# In[74]:


ax=rainfall.groupby(['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC'])['monthly average'].max().sort_values().tail(100).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('month',fontsize=30)
plt.ylabel('monthly average rainfall (in mm)',fontsize=20)
plt.title('month with Maximum rainfall ',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# # visualizing landslide data
# # 
# 

# In[75]:



subdivision = landslide['month'].unique()
subdivision


# In[76]:


subdivision = landslide['year'].unique()
subdivision


# In[77]:


plt.style.use('ggplot')

fig = plt.figure(figsize=(16, 8))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='year', y='sum', data=landslide)
ax = plt.title('Annual landslide')

ax = plt.subplot(2,1,2)
ax = plt.xticks(rotation=90)
ax = sns.barplot(x='year', y='sum', data=landslide)


# In[121]:


plt.style.use('ggplot')

fig = plt.figure(figsize=(20, 10))
ax = plt.subplot(2,1,1)
ax = plt.xticks(rotation=90)
ax = sns.boxplot(x='month', y='monthly average', data=landslide)
ax = plt.title('monthly average landslide')

ax = plt.subplot(2,1,2)
ax = plt.xticks(rotation=90)
ax = sns.barplot(x='month', y='monthly average', data=landslide)


# ## heat map for landslide data

# In[79]:


fig=plt.gcf()
fig.set_size_inches(15,15)
fig=sns.heatmap(landslide.corr(),annot=True,cmap='summer',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# ## data graphically

# In[80]:


ax=landslide.groupby(['year'])['sum'].max().sort_values().tail(10).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))

plt.xlabel('year',fontsize=30)
plt.ylabel('Annual Rainfall (in mm)',fontsize=20)
plt.title('year with Maximum landslide ',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# In[81]:


# #ax=landslide.groupby(['jan', 'feb', 'mar', 'apr','may', 'jun', 'aug', 'sep', 'oct','nov','dec'])['monthly average'].max().sort_values().tail(100).plot.bar(width=0.5,edgecolor='k',align='center',linewidth=4,figsize=(16,8))
# 
# plt.xlabel('month',fontsize=30)
# plt.ylabel('monthly average landslide (in mm)',fontsize=20)
# plt.title('month with Maximum landslide ',fontsize=25)
# ax.tick_params(labelsize=20)
# plt.grid()
# plt.ioff()
# 

# landslide.groupby(['jan', 'feb', 'mar', 'apr','may', 'jun', 'aug', 'sep', 'oct','nov','dec']).mean()['monthly average'].sort_values(ascending=False).head(10)

# Extracting  landslide data for selected GN division
# 


# # landslide data for selected region
# ### LOAD IMAGE
# 

# In[82]:


#extracted data

img=np.array(Image.open('C:/Users/Dilini Dissanayaka/Desktop/Dataset\landslidemap.jpeg'))
fig=plt.figure(figsize=(10,10))
plt.imshow(img,interpolation='bilinear')
plt.axis('off')
plt.ioff()
plt.show()


# In[83]:


landslide=pd.read_excel('C:/Users/Dilini dissanayaka/Desktop/dataset\gnlandslide.xlsx')


# In[84]:


landslide.head()


# In[85]:



landslide['DSD'].unique()


# In[86]:


landslide['GND'].unique()


# In[89]:


ax=landslide.groupby(['DSD'])['year'].max().sort_values().tail(10).plot.bar(width=0.3,edgecolor='k',align='center',linewidth=3,figsize=(8,4))

plt.xlabel('station',fontsize=30)
plt.ylabel('Annual landslide (in mm)',fontsize=20)
plt.title('station with Maximum landslide ',fontsize=25)
ax.tick_params(labelsize=20)
plt.grid()
plt.ioff()


# In[90]:



landslide.groupby('DSD')['Date of Occurrence'].count()


# In[91]:


landslide.groupby('GND')['Date of Occurrence'].count()


# In[112]:


landslide.describe()


# In[113]:


landslide.head()


# #  Time Series Analysis
#  

# ## applying sarima model 
# 

# In[92]:


#rainfall is seasonally distributed sarima model is apply to check seasonal variation
rainfall


# In[93]:


#rainfall['station_name'].unique()


# In[94]:


#rainfall.groupby('station_name')['annual'].count()


# In[95]:


#rainfall.groupby('station_name')['monthly average'].count()


# In[96]:


plt.figure(figsize=(30,15))
sns.lineplot(x=rainfall.index, y=rainfall['monthly average'])
plt.title('rainfall variation')
plt.show()


# In[97]:


y=rainfall['monthly average']


# In[98]:


y.plot()


# In[99]:


y_train=y[:len(y)-11]
y_test=y[(len(y)-11):]


# In[100]:


y_train[-2:]


# In[101]:


y_test[-2:]


# In[102]:


y_train.plot()


# In[103]:



y_test.plot()


# In[104]:


fig,ax=plt.subplots(2,figsize=(20,10))
ax[0]=plot_acf(y_train ,ax=ax[0],lags=10)
ax[1]=plot_pacf(y_train ,ax=ax[1],lags=10)


# In[105]:



#ts_decomp=sm.tsa.seasonal_decompose(y_train,model='additive')
#ts_decomp.plot()
#plt.show


# In[106]:



# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))


# In[107]:


metric_aic_dict=dict()
for pm in pdq:
    for pm_seasonal in seasonal_pdq:
        try:
            model = sm.tsa.statespace.SARIMAX(y_train,
                                            order=pm,
                                            seasonal_order=pm_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            
            model_aic = model.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(pm, pm_seasonal, model_aic.aic))
            metric_aic_dict.update({(pm,pm_seasonal):model_aic.aic})
        except:
            continue


# In[108]:



{k: v for k, v in sorted(metric_aic_dict.items(),key=lambda x:x[1])}


# 
# ## fitting the final model as per the lowest aic

# In[109]:


model = sm.tsa.statespace.SARIMAX(y_train,
                                  order=(0,0,0),
                                  seasonal_order=(0,0,0,12),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)

model_aic=model.fit()
print(model_aic.summary().tables[1])


# In[110]:



model_aic.plot_diagnostics(figsize=(20,10))
plt.show()


# #calculating the rmse
# forecast=model_aic.get_prediction(start=pd.to_datetime('2000'),dynamic=False)
# predictions=forecast.predicted_mean
# 
# actual=y_test['2000':]
# rmse=np.sqrt((predictions - actual)**2).mean()
# print('the root mean squareroot of forecast is {} '.format(round(rmse,2)))
# 


# In[111]:


forecast=model_aic.get_forecast(steps=12)

predictions=forecast.predicted_mean
ci=forecast.conf_int()

#observed plot

fig=y.plot(label='observed',figsize=(14,7))
fig.set_xlabel('Date')
fig.set_ylabel('rainfall')
fig.fill_between(ci.index,
                ci.iloc[:,0],
                ci.iloc[:,1],color='k',alpha=.2)

#prediction
predictions.plot(ax=fig,label='predictions',alpha=7,figsize=(20,10))

plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




