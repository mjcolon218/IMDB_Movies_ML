![screenshot](images/imdb2.jpg?raw=True)

# IMDB Movies analysist and Revenue Prediction

# Overview
This project involves analyzing the IMDB Movie Dataset to predict movie revenues. The dataset includes various features such as movie ratings, genres, director, actors, and revenues. The goal is to build machine learning models to understand and predict the revenue of movies based on these features.

* Data Cleaning: Handled missing values, particularly in the 'Revenue (Millions)' column.
* Feature Engineering:
Extracted the lead actor from the 'Actors' column.
Utilized features like 'Rating', 'Votes', 'Runtime (Minutes)', 'Genre', 'Director', and 'Lead Actor'.
Applied one-hot encoding to categorical features and standardization to numerical features.
* Data Transformation and Splitting:
Transformed the data using a ColumnTransformer.
Split the data into training and testing sets.

# Exploratory Data Analysis (EDA)
## Several visualizations were created to understand the data better:

* Distribution of movies over years.
* Distribution of movie ratings.
* Distribution of movie revenues.
* Rating vs. revenue with year as a hue.
* Bar plots showing the count of movies in each genre.
* Word Cloud for movie titles.
* Pie Chart for movie distribution by year.
# Basic Statistics
* Count: There are 838 movies in the dataset.
* Year: The movies range from 2006 to 2016.
* Runtime: The average runtime is approximately 115 minutes.
* Rating: The average IMDb rating is around 6.8, with a minimum of 1.9 and a maximum of 9.0.
* Votes: There's a wide range in the number of votes, indicating varying levels of popularity.
* Revenue: The average revenue is about 84.56 million dollars.
* Metascore: The average metascore is around 59.6.
```python
# Basic statistics
basic_stats = movie_data.describe()
basic_stats
```
<div>
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>Rank</th>
      <th>Year</th>
      <th>Runtime (Minutes)</th>
      <th>Rating</th>
      <th>Votes</th>
      <th>Revenue (Millions)</th>
      <th>Metascore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>838.000000</td>
      <td>838.000000</td>
      <td>838.00000</td>
      <td>838.000000</td>
      <td>838.000000</td>
      <td>8.380000e+02</td>
      <td>838.000000</td>
      <td>838.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>484.247017</td>
      <td>485.247017</td>
      <td>2012.50716</td>
      <td>114.638425</td>
      <td>6.814320</td>
      <td>1.932303e+05</td>
      <td>84.564558</td>
      <td>59.575179</td>
    </tr>
    <tr>
      <th>std</th>
      <td>286.572065</td>
      <td>286.572065</td>
      <td>3.17236</td>
      <td>18.470922</td>
      <td>0.877754</td>
      <td>1.930990e+05</td>
      <td>104.520227</td>
      <td>16.952416</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.00000</td>
      <td>66.000000</td>
      <td>1.900000</td>
      <td>1.780000e+02</td>
      <td>0.000000</td>
      <td>11.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>237.250000</td>
      <td>238.250000</td>
      <td>2010.00000</td>
      <td>101.000000</td>
      <td>6.300000</td>
      <td>6.127650e+04</td>
      <td>13.967500</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>474.500000</td>
      <td>475.500000</td>
      <td>2013.00000</td>
      <td>112.000000</td>
      <td>6.900000</td>
      <td>1.368795e+05</td>
      <td>48.150000</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>728.750000</td>
      <td>729.750000</td>
      <td>2015.00000</td>
      <td>124.000000</td>
      <td>7.500000</td>
      <td>2.710830e+05</td>
      <td>116.800000</td>
      <td>72.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>999.000000</td>
      <td>1000.000000</td>
      <td>2016.00000</td>
      <td>187.000000</td>
      <td>9.000000</td>
      <td>1.791916e+06</td>
      <td>936.630000</td>
      <td>100.000000</td>
    </tr>
  </tbody>
</table>
</div>

![screenshot](images/histograms.png?raw=True)
* Distribution of Movies Over Years (Histogram):

* Showed an increasing trend in the number of movies over the years, peaking around 2016.
* Inference: Indicates a growth in movie production or improved data collection in recent years.
* Distribution of Movie Ratings (Histogram):

* Revealed a somewhat normal but slightly left-skewed distribution, with most ratings between 6 and 8.
* Inference: Suggests a general trend of average to good quality movies in the dataset.
![screenshot](images/wordcloud.png?raw=True)
![screenshot](images/ratingvrevenue.png?raw=True)
![screenshot](images/barcharts.png?raw=True)
* How I handled and grouped all the Genres accordingly.
```python
# Function to categorize movies into broader genre groups
def categorize_genre(genres):
    # Define genre groups
    groups = {
        "Drama/Thriller/Crime": ['Drama', 'Thriller', 'Crime'],
        "Comedy/Romance": ['Comedy', 'Romance'],
        "Action/Adventure/Sci-Fi/Fantasy": ['Action', 'Adventure', 'Sci-Fi', 'Fantasy'],
        "Family/Animation": ['Family', 'Animation'],
        "Horror/Mystery": ['Horror', 'Mystery'],
        "Biography/History/Music/Sport/War": ['Biography', 'History', 'Music', 'Sport', 'War']
    }
    
    for group, genres_list in groups.items():
        if any(genre in genres for genre in genres_list):
            return group
    return "Other"

# Apply the function to the dataset
movie_data['Broad Genre'] = movie_data['Genre'].apply(lambda x: categorize_genre(x.split(',')))

# Displaying the new categorization
movie_data[['Title', 'Genre', 'Broad Genre']].head()

```
###### Distribution of Movies Over Years:
* The histogram shows a roughly increasing trend in the number of movies over the years, with a peak around 2016. This indicates a growth in movie production over the analyzed period.
* Distribution of Movie Ratings:

* The distribution of movie ratings is somewhat normal but slightly left-skewed. Most movies have ratings between 6 and 8. Very few movies have ratings below 4 or above 8.
Distribution of Movie Revenues:

* The revenue distribution is right-skewed, indicating that a large number of movies earn relatively low revenues, while a few movies earn significantly high revenues.
# Insights and Observations
* Popularity and Quality: The wide range of votes and ratings suggests a diverse mix of popularity and perceived quality among the movies.
* Revenue Distribution: The skewness in the revenue distribution might indicate that only a small number of movies are highly successful commercially.
* Time Trend: The increasing number of movies over the years could be reflective of the growing film industry or possibly increased data collection in recent years.

# Feature Engineering/ Model

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


# Load the dataset
movie_data = pd.read_csv('data/IMDB-Movie-Data.csv')

# Data Cleaning
# Handling missing values in 'Revenue (Millions)'
movie_data_clean = movie_data.dropna(subset=['Revenue (Millions)'])

# Feature Engineering
# Simplifying the actors column - taking only the first actor
movie_data_clean['Lead Actor'] = movie_data_clean['Actors'].apply(lambda x: x.split(',')[0])

# Selecting features for the model
features = ['Rating', 'Votes', 'Runtime (Minutes)', 'Genre', 'Director', 'Lead Actor']
target = 'Revenue (Millions)'

# Data and target
X = movie_data_clean[features]
y = movie_data_clean[target]

# Defining categorical and numerical features
categorical_features = ['Genre', 'Director', 'Lead Actor']
numerical_features = ['Rating', 'Votes', 'Runtime (Minutes)']

# Creating transformers
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Data Transformation
X_transformed = preprocessor.fit_transform(X)

# Splitting the Dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)
# Initialize models
lr_model = LinearRegression()
rf_model = RandomForestRegressor(random_state=42)
gb_model = GradientBoostingRegressor(random_state=42)

# Training and Evaluating Linear Regression
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_rmse = np.sqrt(lr_mse)

# Training and Evaluating Random Forest
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_rmse = np.sqrt(rf_mse)

# Training and Evaluating Gradient Boosting
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)
gb_mse = mean_squared_error(y_test, gb_predictions)
gb_rmse = np.sqrt(gb_mse)

# Print the scores
print("Linear Regression RMSE:", lr_rmse)
print("Random Forest RMSE:", rf_rmse)
print("Gradient Boosting RMSE:", gb_rmse)
Linear Regression RMSE: 120.34120663577627
Random Forest RMSE: 82.70873820502295
Gradient Boosting RMSE: 77.83762986871209

```
![screenshot](images/regressionmodel.png?raw=True)
![screenshot](images/randomforest.png?raw=True)
![screenshot](images/gradientboosting.png?raw=True)

# General Observations:
* Linear Regression Coefficients: The visualization helps in understanding the direct influence of each feature on movie revenues. Features with high absolute coefficient values are most influential.

* Random Forest and Gradient Boosting Feature Importances: These visualizations provide insights into the features that most significantly impact the model's decision-making process. Unlike Linear Regression, these importances do not directly imply the direction (positive or negative) of the impact on revenue.

* The choice of model depends on the dataset's complexity and the nature of relationships between features and the target variable.
Ensemble models like Random Forest and Gradient Boosting are typically more powerful for capturing complex patterns but may require more computational resources and careful tuning.
The top features identified by each model can guide business strategies, such as focusing on specific genres, directors, or actors to maximize revenue.
The differing results between models underscore the importance of trying multiple modeling approaches and comparing their results to get a comprehensive view.
# Conclusion:
* The analysis of the IMDB movie dataset using these models provides valuable insights into factors influencing movie revenues. By comparing the results of different models, you can gain a deeper understanding of the underlying patterns and relationships within the data. These insights can inform decision-making in the movie industry, from production to marketing strategies.
