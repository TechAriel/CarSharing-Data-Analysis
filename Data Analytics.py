#!/usr/bin/env python
# coding: utf-8

# 1. Import the CarSharing table into a CSV file and preprocess it with python. You need to
# drop duplicate rows and deal with null values using appropriate methods. 

# In[1]:


import pandas as pd


# Load the CSV file into a DataFrame
car_sharing_df = pd.read_csv("CarSharing.csv")


# In[2]:


# Drop duplicate rows
car_sharing_df_cleaned = car_sharing_df.drop_duplicates()


# In[3]:


#check for columns with null values
print(car_sharing_df_cleaned.isnull().sum())


# Results
# id               0
# timestamp        0
# season           0
# holiday          0
# workingday       0
# weather          0
# temp          1202
# temp_feel      102
# humidity        39
# windspeed      200
# demand           0
# dtype: int64

# In[4]:


# This code was used to decide whether to use mean or median for filling null values
distribution_assessment = car_sharing_df_cleaned[['temp', 'temp_feel', 'humidity', 'windspeed']].describe()
print(distribution_assessment)


# results:
# 
#              temp    temp_feel    humidity    windspeed
# count  7506.000000  8606.000000  8669.00000  8508.000000
# mean     20.089454    23.531261    60.99354    13.048589
# std       8.023304     8.737997    19.67989     8.311058
# min       0.820000     0.760000     0.00000     0.000000
# 25%      13.940000    15.910000    46.00000     7.001500
# 50%      20.500000    24.240000    60.00000    12.998000
# 75%      26.240000    31.060000    77.00000    19.001200
# max      41.000000    45.455000   100.00000    56.996900

# In[5]:


# Filling null values with the median of their respective columns
car_sharing_df_cleaned['temp'].fillna(car_sharing_df_cleaned['temp'].median(), inplace=True)
car_sharing_df_cleaned['temp_feel'].fillna(car_sharing_df_cleaned['temp_feel'].median(), inplace=True)
car_sharing_df_cleaned['humidity'].fillna(car_sharing_df_cleaned['humidity'].median(), inplace=True)
car_sharing_df_cleaned['windspeed'].fillna(car_sharing_df_cleaned['windspeed'].median(), inplace=True)


# In[6]:


# Verifying the dataset for null values
null_values_check = car_sharing_df_cleaned.isnull().sum()
print(null_values_check)


# Results:
# id            0
# timestamp     0
# season        0
# holiday       0
# workingday    0
# weather       0
# temp          0
# temp_feel     0
# humidity      0
# windspeed     0
# demand        0
# dtype: int64

# In[ ]:





# 2. Using appropriate hypothesis testing, determine if there is a significant relationship
# between each column (except the timestamp column) and the demand rate. Report the
# tests’ results. 

# In[7]:


from scipy.stats import pearsonr, f_oneway
import numpy as np

# Pearson's correlation for numerical columns
pearson_results = {}
for column in ['temp', 'temp_feel', 'humidity', 'windspeed']:
    correlation, p_value = pearsonr(car_sharing_df_cleaned[column], car_sharing_df_cleaned['demand'])
    pearson_results[column] = {'correlation': correlation, 'p_value': p_value}

# One-way ANOVA for categorical columns
anova_results = {}
for column in ['season', 'holiday', 'workingday', 'weather']:
    categories = car_sharing_df_cleaned[column].unique()
    groups = [car_sharing_df_cleaned['demand'][car_sharing_df_cleaned[column] == category] for category in categories]
    f_stat, p_value = f_oneway(*groups)
    anova_results[column] = {'f_stat': f_stat, 'p_value': p_value}

pearson_results, anova_results


# Results:
# ({'temp': {'correlation': 0.36914992862076795,
#    'p_value': 2.5959578343747293e-279},
#   'temp_feel': {'correlation': 0.3902014765082438,
#    'p_value': 1.186150842e-314},
#   'humidity': {'correlation': -0.33079479771176457,
#    'p_value': 2.1248021477972816e-221},
#   'windspeed': {'correlation': 0.11831871828095224,
#    'p_value': 1.5960141054487996e-28}},
#  {'season': {'f_stat': 150.0648218917323, 'p_value': 8.024921568562112e-95},
#   'holiday': {'f_stat': 0.011054437013371764, 'p_value': 0.9162670761960895},
#   'workingday': {'f_stat': 2.868367223957404, 'p_value': 0.09037224773166397},
#   'weather': {'f_stat': 48.586185236529495,
#    'p_value': 3.9279297308870713e-31}})

# In[ ]:





# 3. Please describe if you see any seasonal or cyclic pattern in the temp, humidity, windspeed,
# or demand data in 2017. Describe your answers

# In[8]:


import matplotlib.pyplot as plt

# Convert the 'timestamp' column to datetime format
car_sharing_df_cleaned['timestamp'] = pd.to_datetime(car_sharing_df_cleaned['timestamp'])

# Assuming the dataset only contains data for 2017 based on the initial observations,
# but this ensures we're only looking at 2017 if there are other years
car_sharing_df_cleaned_2017 = car_sharing_df_cleaned[car_sharing_df_cleaned['timestamp'].dt.year == 2017]

# Set the timestamp as the index
car_sharing_df_cleaned_2017.set_index('timestamp', inplace=True)

# Plotting
fig, axs = plt.subplots(4, 1, figsize=(14, 18), sharex=True)

# Temperature plot
axs[0].plot(car_sharing_df_cleaned_2017.index, car_sharing_df_cleaned_2017['temp'], label='Temperature', color='orange')
axs[0].set_ylabel('Temperature (°C)')
axs[0].set_title('Temperature over 2017')
axs[0].legend()

# Humidity plot
axs[1].plot(car_sharing_df_cleaned_2017.index, car_sharing_df_cleaned_2017['humidity'], label='Humidity', color='blue')
axs[1].set_ylabel('Humidity (%)')
axs[1].set_title('Humidity over 2017')
axs[1].legend()

# Windspeed plot
axs[2].plot(car_sharing_df_cleaned_2017.index, car_sharing_df_cleaned_2017['windspeed'], label='Windspeed', color='green')
axs[2].set_ylabel('Windspeed (km/h)')
axs[2].set_title('Windspeed over 2017')
axs[2].legend()

# Demand plot
axs[3].plot(car_sharing_df_cleaned_2017.index, car_sharing_df_cleaned_2017['demand'], label='Demand', color='red')
axs[3].set_ylabel('Demand')
axs[3].set_title('Demand over 2017')
axs[3].legend()

plt.xlabel('Date')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:






# 4. Use an ARIMA model to predict the weekly average demand rate. Consider 30 percent of
# data for testing. 

# In[9]:


# Aggregate data to weekly averages
weekly_demand = car_sharing_df_cleaned_2017['demand'].resample('W').mean()

# Split data into training and testing sets
split_index = int(len(weekly_demand) * 0.7)
train_data, test_data = weekly_demand[:split_index], weekly_demand[split_index:]

# Check the lengths and first rows of both sets to confirm
train_data.shape, test_data.shape, train_data.head(), test_data.head()


# Results:
# ((36,),
#  (16,),
#  timestamp
#  2017-01-01    3.123272
#  2017-01-08    3.437559
#  2017-01-15    3.364685
#  2017-01-22    3.415826
#  2017-01-29         NaN
#  Freq: W-SUN, Name: demand, dtype: float64,
#  timestamp
#  2017-09-10    4.253224
#  2017-09-17    4.707347
#  2017-09-24    4.635734
#  2017-10-01    4.288362
#  2017-10-08    4.608928
#  Freq: W-SUN, Name: demand, dtype: float64)

# In[10]:


from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Fill missing values using forward fill
train_data_filled = train_data.fillna(method='ffill')

# Fill missing values in test data using forward fill
test_data_filled = test_data.fillna(method='ffill')

# Define and fit the ARIMA model
arima_model = ARIMA(train_data_filled, order=(1, 1, 1))
arima_result = arima_model.fit()

# Predictions
predictions = arima_result.forecast(steps=len(test_data))

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test_data_filled, predictions))

rmse, predictions.head()



# Results:
# (0.37787366676432993,
#  2017-09-10    4.787714
#  2017-09-17    4.781056
#  2017-09-24    4.779280
#  2017-10-01    4.778807
#  2017-10-08    4.778680
#  Freq: W-SUN, Name: predicted_mean, dtype: float64)

# In[ ]:





# 5. Use a random forest regressor and a deep neural network to predict the demand rate and
# report the minimum square error for each model. Which one is working better? Why?
# Please describe the reason. 

# In[11]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Preparing the dataset for modeling (excluding categorical variables for simplicity)
features = car_sharing_df_cleaned_2017[['temp', 'temp_feel', 'humidity', 'windspeed']].fillna(method='ffill')
target = car_sharing_df_cleaned_2017['demand'].fillna(method='ffill')

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)

# Scaling features for the DNN model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Deep Neural Network
dnn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

dnn_model.compile(optimizer=Adam(), loss='mean_squared_error')
dnn_model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)
dnn_predictions = dnn_model.predict(X_test_scaled).flatten()
dnn_mse = mean_squared_error(y_test, dnn_predictions)

rf_mse, dnn_mse


# Results:
# 51/51 [==============================] - 0s 861us/step
# (1.8315273704037978, 1.4613300387590198)

# In[ ]:





# 6. Categorize the demand rate into the following two groups: demand rates greater than the
# average demand rate and demand rates less than the average demand rate. Use labels 1
# and 2 for the first and the second groups, respectively. Now, use three different classifiers
# to predict the demand rates’ labels and report the accuracy of all models. Use 30 percent
# of data for testing. 

# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Assuming 'df' is your DataFrame
average_demand = car_sharing_df_cleaned['demand'].mean()
car_sharing_df_cleaned['demand_label'] = car_sharing_df_cleaned['demand'].apply(lambda x: 1 if x > average_demand else 2)


# In[13]:


# Define features and target
X = car_sharing_df_cleaned[['temp', 'temp_feel', 'humidity', 'windspeed']]  # example features
y = car_sharing_df_cleaned['demand_label']

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Optional: Scale features (Especially important for Logistic Regression)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[14]:


logistic_model = LogisticRegression()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)


# In[15]:


# Logistic Regression
logistic_model.fit(X_train_scaled, y_train)
logistic_pred = logistic_model.predict(X_test_scaled)
logistic_accuracy = accuracy_score(y_test, logistic_pred)

# Random Forest Classifier
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

# Gradient Boosting Classifier
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print(f"Logistic Regression Accuracy: {logistic_accuracy}")
print(f"Random Forest Accuracy: {rf_accuracy}")
print(f"Gradient Boosting Accuracy: {gb_accuracy}")


# Results:
# Logistic Regression Accuracy: 0.7206276310753923
# Random Forest Accuracy: 0.689628779181018
# Gradient Boosting Accuracy: 0.7317259854573287

# In[ ]:





# 7. Assume k is the number of clusters. Set k=2, 3, 4, and 12 and use 2 methods to cluster
# the temp data in 2017. Which k gives the most uniform clusters? (Clusters are called
# uniform when the number of samples falling into each cluster is close.) 

# In[16]:


from sklearn.cluster import KMeans, AgglomerativeClustering
import numpy as np

# Assuming 'temp_data' contains the temperature data for 2017
temp_data = car_sharing_df_cleaned_2017['temp'].values.reshape(-1, 1)

ks = [2, 3, 4, 12]

def evaluate_uniformity(labels):
    unique, counts = np.unique(labels, return_counts=True)
    return np.var(counts)

for k in ks:
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(temp_data)
    kmeans_variance = evaluate_uniformity(kmeans_labels)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=k)
    hierarchical_labels = hierarchical.fit_predict(temp_data)
    hierarchical_variance = evaluate_uniformity(hierarchical_labels)
    
    print(f"k={k}: K-Means Variance = {kmeans_variance}, Hierarchical Variance = {hierarchical_variance}")


# Result:
# k=2: K-Means Variance = 385641.0, Hierarchical Variance = 781456.0
# k=3: K-Means Variance = 108955.55555555555, Hierarchical Variance = 11373.555555555555
# k=4: K-Means Variance = 195621.25, Hierarchical Variance = 225694.25
# k=12: K-Means Variance = 51482.305555555555, Hierarchical Variance = 52667.97222222224
