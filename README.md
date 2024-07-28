# Woodpecker_round_2

Forecasting for Inventory management


## Energy

Every day, power grid operators need to predict how much electricity people will use in the next 24 hours. They do this by looking at past energy use data, considering factors like the day of the week, holidays, and even the weather. This forecast, broken down into hourly slices, helps them plan how much power to generate and buy ahead of time. It's crucial for keeping the electricity grid running smoothly and avoiding shortages.


### Methods Used
* Data Wrangling
* Machine Learning
* Regression
* Neural Networks
* Predictive Modelling
* Walk forward cross validation
* Hypothesis Testing
  ### Technologies
* Python
* Keras, Tensorflow
* Pandas, Numpy, Jupyter
* Statsmodels, Scikit-Learn
* Prophet
* Joblib, holidays libraries
* Google Cloud Platform
Time series forecasting is the task of predicting future values in a time-ordered sequence of observations. It is a common problem and has applications in many industries. This example focuses on energy demand forecasting where the goal is to predict the future load on an energy grid. It is a critical business operation for companies in the energy sector.

Energy data was obtained from the [ENTSOE Transparency Platform](https://transparency.entsoe.eu/). The platform provides electrical demand, generation, and price information for all european countries since 2015. Data is available at hourly and daily timeframes. 

Weather data was purchased from [OpenWeatherApi](https://openweathermap.org/api). Data from the five largest cities in spain was pruchased for the previous 8 years. Data includes hourly measurements of temperature (min, max), humidity, preciptation (1h, 3h), snow (1h, 3h), and general descrption of weather status (text format)

Day of the week and holiday data was genreated using the holidays library. The [following notebook](ENERGY/data_creation_day_types.ipynb) explains how the helper function was implemented. The helper function can be accessed from [make_holidays_data.py](ENERGY/make_holidays_data.py).

### Modelling Short-Term Energy Demand

Features used to generate forecasts include autocorrelated hourly energy consumption, hourly weather data, days of the week, and holidays. A detailed decrption of each feature is below:

- Energy demand lags ranging between 7 days (168 hours) and 1 month
- Two PCA vectors generated from weather data
- Days of the week one hot encoded
- Holidays one hot encoded

  
## Project needs and core tasks

- data processing/cleaning
    - cleaning of energy data, weather data, and generation of holiday data
    - process data to generate autoregressive features
    - processing data to frame the problem for SARIMA, Prophet, LSTM
- data exploration
    - visualize energy consumption patterns at different temporal scales
    - visualize weather correlations and 
- statistical modeling
    - (auto)correlation analysis of model features and feature selection
    - PCA transformation of colinear (weather) features
    - parameter selection for SARIMA model: determining differncing, seasonality, and trend components
    - parameter selection for Prophet model: configure base mode, additional regressors
- machine learning
    - configuration, hyperparmeter tuning, training, and testing of LSTM neural network
- data pipeline
    - helper functions to prepare input, calcualte erros, run walk forward cross validation, and produce visualizations for each model
- reporting/presentation
    - documentation of helper functions
    - presentation of work at live event
 
      

## E-commerce

This focuses on analyzing eCommerce invoice data to derive insights and develop predictive models for retail business optimization. It encompasses data collection through web scraping, data processing, analysis, and deployment of machine learning models.

## Methods Used
* Data Wrangling
* Machine Learning
* Time Series Analysis
* Regression
* Neural Networks
* Predictive Modelling
* Walk-forward Cross Validation
* Hypothesis Testing
* Web Scraping

## Technologies
* Python
* Docker
* Apache Airflow
* PostgreSQL
* Terraform
* Pandas, NumPy, Jupyter
* Scikit-Learn, TensorFlow, Keras
* Power BI
* Google Cloud Platform (or AWS, based on your cloud preference)

Time series forecasting is a crucial task in retail, particularly for inventory management and sales prediction. This project focuses on eCommerce invoice analysis, aiming to predict future sales, optimize inventory levels, and derive actionable insights for business operations.

## Modelling eCommerce Sales and Inventory
Features used for forecasting include:
- Historical sales data with lags ranging from 7 days to 3 months
- Weather data transformed into PCA vectors
- One-hot encoded days of the week
- One-hot encoded holidays
- Promotional event flags
- Seasonal indices

## Project Needs and Core Tasks
- Data Collection and Processing
  - Web scraping of eCommerce invoice data
  - Cleaning and preprocessing of sales, weather, and promotional data
  - Generation of time-based features (holidays, seasons)
  - Creation of autoregressive features for time series models
- Data Exploration and Visualization
  - Analysis of sales patterns across different product categories
  - Visualization of seasonal trends and promotional impacts
  - Correlation analysis between sales and external factors (weather, events)
- Statistical Modeling and Machine Learning
  - Time series decomposition to identify trends and seasonality
  - Implementation of ARIMA, Prophet, and LSTM models for sales forecasting
  - Feature selection through correlation analysis and dimensionality reduction
  - Hyperparameter tuning for optimal model performance
- Data Pipeline and Infrastructure
  - Development of an ETL pipeline using Apache Airflow
  - Setup of PostgreSQL databases for data storage and retrieval
  - Implementation of cloud infrastructure using Terraform
  - Containerization of the application components with Docker
- Business Intelligence and Reporting
  - Creation of interactive dashboards using Power BI
  - Development of real-time reporting mechanisms for inventory alerts
  - Documentation of data pipelines and model architectures
- Deployment and Monitoring
  - Continuous integration and deployment setup
  - Implementation of model performance monitoring and retraining schedules

This combines data engineering, machine learning, and business intelligence to provide a comprehensive solution for eCommerce inventory management and sales forecasting. The modular structure allows for easy scaling and adaptation to changing business needs.

## Retail

The COVID-19 pandemic accelerated the shift from traditional to online shopping, especially in developed economies. This surge in e-commerce has created new challenges and opportunities for businesses, particularly in demand forecasting.

This study aimed to assess the value of incorporating Google Trends data into sales forecasting models. To do this, we compared the performance of SARIMA, Prophet, XGBoost, and LSTM models using e-commerce datasets. We found that while adding Google Trends data didn't significantly improve overall forecast accuracy, it did enhance predictions for specific product categories.

Our results suggest that improving forecast precision can positively impact inventory management. However, the exact impact on inventory performance requires further investigation using simulation tools. This research contributes to the ongoing exploration of machine learning techniques in sales forecasting.

### Prediction Task  

- The prediction task for the Brazilian e-commerce dataset is to forecast the weekly number of sales transactions by product category. The scope of sales transactions from the e-commerce dataset are limited to the top 7 selling product categories. Thus, the e-commerce dataset is split into 7 separate datasets and each forecasting model (SARIMA, FBProphet, XGBoost, LSTM) is trained and tested 7 times, once for each product category

- The prediction task for Breakfast at the Frat dataset is to forecast the weekly number of units sold of 4 items across 3 stores. Therefore, the Breakfast at the Frat dataset is split into 12 separate datasets and each forecasting model (SARIMA, FBProphet, XGBoost, LSTM) is trained and tested 12 times, once for each product and store combination. The data used from the Breakfast at the Frat dataset include sales history, promotional, product, manufacturer and store information

### Experiment Setup 

- The experiment utilizes configuration and parameter files to pre-process data and determine parameter values required to run the forecasting models. 

<div align="center">
  Experiment Conceptual Diagram
</div>

|![ExperimentDesign_MSc](https://user-images.githubusercontent.com/39706513/101991375-4bc7e400-3c7a-11eb-968f-00dbf5d85617.png) | 
|:--:|
| *[MLflow](https://mlflow.org/docs/latest/tracking.html) is used to track parameters and performance metrics* |

##### model parameters

For the XGBoost model, there are a few parameters that need to be specified:

- `window_size` is used to create the number of lagged values as input for the model. A value of 52 will create 52 lags of the target column. 
- `avg_units` is used to create rolling averages using the lag-1 column to avoid leakage. It represents a list of rolling average features. A value of 2 will create a two time steps rolling-average, while a value of 16 will create a rolling-average of 16 time steps. Each will represent a column in the dataset.
- `gtrends_window_size` is used to cerate the number of lagged values for the google trends series. Each google trend series will be created usign this value.
- `search_iter` is used to specify how many rounds of hyperparameter search to perform using Hyperopt.

For the LSTM model, there are different parameters that need to be specified:

- `window_size` is used to create the number of lagged values as input for the model. A value of 52 will create 52 lags of the target column. 
- `gtrends_window_size` is used to cerate the number of lagged values for the google trends series. This value must be the same as the window_size as the LSTM expects the same number of dimensions for each feature.
- `dropout` is a hyperparameter that is specified in advance. It is used to reduce overfitting but could be included in the hyperparameter search if desired.
- `units_strategy` is used to determine how to select the number of hidden units for each LSTM layer. A stable strategy will keep the number of units constant accross layers. A decrease strategy will halve the number of units per layer. For a three layer model with initial number of units set to 50, the stable strategy will assign 50 units for each layer while the decrease strategy will set 50 for the first layer, 25 for the second layer and 16 for the third layer.
- `optimizers` the optimizer to use. 
- `loss` the loss to use.
- `search_iter` the number of random search iterations to perform during hyperparameter tunning.
