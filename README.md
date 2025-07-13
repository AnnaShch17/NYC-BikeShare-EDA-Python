# New York City Bike Share EDA

In this project, I conducted Exploratory Data Analysis (EDA) on the New York City Bike Share dataset using Python. 
[This dataset](https://www.kaggle.com/datasets/akkithetechie/new-york-city-bike-share-dataset) can be found on Kaggle.
My analysis focused on understanding trip duration patterns, users' demographics, and data quality issues. 
I applied data cleaning techniques, statistical analysis, and visualizations (using Matplotlib and Seaborn) to uncover actionable insights.

I began the analysis with descriptive statistics such as the mean, median, mode, standard deviation, and range to summarize trip durations. 
The data showed that the majority of trips were relatively short, typically between 4 and 7 minutes. However, the presence of extreme 
outliers – some trips last over several days or even months – indicated potential data quality issues, such as input errors or unreturned bikes.

To address data quality issues, I used the Interquartile Range (IQR) method to identify and remove outliers. This helped reduce skewness and made
the dataset look more representative of typical usage patterns. I also used histograms and box plots to visualize how the distribution
of trip durations changed before and after cleaning.

I also explored the following:<br>

•	How trip frequency varies by user type (Subscriber vs Customer).<br>
•	Differences in usage patterns by gender.<br>
•	Time-based patterns (e.g., peak trip hours or popular months).<br>
•	Correlations between variables using heatmaps.

Techniques used:<br>

•	Descriptive statistics.<br>
•	Outlier detection and removal (IQR).<br>
•	Histograms and box plots.<br>
•	Grouped bar charts.<br>
•	Time series trends.<br>
•	Correlation analysis.


The goal of this project was to create a clean and meaningful dataset that reveals how New Yorkers use bike-sharing services. I focused on understanding 
who the users are, when they tend to ride, and how long their trips usually last. These insights support user behavior modeling and data-driven decisions 
in urban planning.
I completed the analysis and visualizations in Python on Kaggle. The notebook can be found here: [NYC Bike Share EDA on Kaggle](https://www.kaggle.com/code/annashcherbinina/eda-new-york-bike-data-anna-shcherbinina).

## Exploring the Dataset

```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

First, let's load data into a DataFrame using pd.read_csv(). This function reads the data from the CSV file located in the specified path and stores it in the variable df. 
Now, df holds all the bike-share data from New York City, collected between 2015 and 2017.

```df= pd.read_csv("../input/new-york-city-bike-share-dataset/NYC-BikeShare-2015-2017-combined.csv")```


Next, I removed a column named 'Unnamed: 0'. This column is likely an unnecessary index column created during data export, so I do not need it. 
I used df.drop() to remove it, specifying axis=1 to drop a column (not a row) and inplace=True to make the change directly to df.

```df.drop("Unnamed: 0",axis=1,inplace =True)```


Finally, I used df.head() to display the first 5 rows of the DataFrame. This gives a quick snapshot of the data, so I can see what it looks like and start understanding the structure of the dataset, such as column names and sample values.

```df.head(5)```



| Trip Duration | Start Time       | Stop Time        | Start Station ID | Start Station Name | Start Lat  | Start Long  | End Station ID | End Station Name | End Lat   | End Long   | Bike ID | User Type   |
|---------------|------------------|------------------|------------------|--------------------|------------|-------------|----------------|------------------|-----------|------------|---------|-------------|
| 376           | 2015-10-01 00:16 | 2015-10-01 00:22 | 3212             | Christ Hospital    | 40.734786  | -74.050444  | 3207           | Oakland Avenue   | 40.737604 | -74.052478 | 24470   | Subscriber  |
| 739           | 2015-10-01 00:27 | 2015-10-01 00:39 | 3207             | Oakland Avenue     | 40.737604  | -74.052478  | 3212           | Christ Hospital  | 40.734786 | -74.050444 | 24481   | Subscriber  |
| 2714          | 2015-10-01 00:32 | 2015-10-01 01:18 | 3193             | Lincoln Park       | 40.724605  | -74.078406  | 3193           | Lincoln Park     | 40.724605 | -74.078406 | 24628   | Subscriber  |
| 275           | 2015-10-01 00:34 | 2015-10-01 00:39 | 3199             | Newport Pkwy       | 40.728745  | -74.032108  | 3187           | Warren St.       | 40.721124 | -74.038051 | 24613   | Subscriber  |
| 561           | 2015-10-01 00:40 | 2015-10-01 00:49 | 3183             | Exchange Place     | 40.716247  | -74.033459  | 3192           | Liberty Rail     | 40.711242 | -74.055701 | 24668   | Customer    |



The command df.info() is used to quickly get an overview of the dataset, including:<br>

• The number of rows and columns.<br>
• The data types of each column (e.g., integer, float, object).<br>
• The number of non-null values in each column, which helps identify missing data. This method provides a quick summary of the dataset's structure and allows 
to spot potential issues, such as columns with missing values or incorrect data types.


```df.info()```




| **#** | **Column**                | **Non-Null Count**   | **Dtype** |
|-------|---------------------------|-----------------------|-----------|
| 0     | Trip Duration             | 735502 non-null       | int64     |
| 1     | Start Time                | 735502 non-null       | object    |
| 2     | Stop Time                 | 735502 non-null       | object    |
| 3     | Start Station ID          | 735502 non-null       | int64     |
| 4     | Start Station Name        | 735502 non-null       | object    |
| 5     | Start Station Latitude    | 735502 non-null       | float64   |
| 6     | Start Station Longitude   | 735502 non-null       | float64   |
| 7     | End Station ID            | 735502 non-null       | int64     |
| 8     | End Station Name          | 735502 non-null       | object    |
| 9     | End Station Latitude      | 735502 non-null       | float64   |
| 10    | End Station Longitude     | 735502 non-null       | float64   |
| 11    | Bike ID                   | 735502 non-null       | int64     |
| 12    | User Type                 | 735502 non-null       | object    |
| 13    | Birth Year                | 735502 non-null       | float64   |
| 14    | Gender                    | 735502 non-null       | int64     |
| 15    | Trip_Duration_in_min      | 735502 non-null       | int64     |





## Let's start EDA.

```df.shape```

(735502, 16)
There are 16 columns and 735502 rows in the dataset.

Let's first check if there are any duplicates. The first step is to clear our dataset from duplicates.

```df.drop_duplicates(inplace=True)```


drop_duplicates(): This function removes duplicate rows from the DataFrame.
inplace=True: This modifies the original DataFrame directly, so tere is no need to reassign it.



```df.shape```

(339620, 16)
After deduplication, there are just 339,620 records in the dataset left.



**Descriptive Statistics**<br>

**Mean:** The average value of a dataset.<br>
**Median:** The middle value when the data is sorted.<br>
**Mode:** The most frequently occurring value.<br>
**Standard Deviation:** Measures how spread out the data is from the mean.<br>
**Range:** The difference between the maximum and minimum values.

```
python mean_duration = df['Trip Duration'].mean()
median_duration = df['Trip Duration'].median()
mode_duration = df['Trip Duration'].mode()[0]
std_dev_duration = df['Trip Duration'].std()  # Standard Deviation
range_durtion = df['Trip Duration'].max() - df['Trip Duration'].min()  # Range
print("Mean:", mean_duration, "Median:", median_duration, "Mode:", mode_duration,"Std Dev:", std_dev_duration, "Range :", range_durtion)
```


Mean: 962.9629203227137 Median: 384.0 Mode: 244 Std Dev: 48685.70128924114 Range : 20260150


**Based on the descriptive statistics of the Trip Duration data:** <br>

**Mean** (962.96 seconds): The average trip duration is about 16 minutes, suggesting that, on average, bike trips are relatively short.<br>
**Median** (384 seconds): The median trip duration is around 6.4 minutes, indicating that half of the bike trips are shorter than this time, showing that most trips are relatively brief.<br>
**Mode** (244 seconds): The most common trip duration is 244 seconds (about 4 minutes), reflecting a large number of quick bike trips.<br>
**Standard Deviation** (48,685.70 seconds): The data shows a high variability in trip durations, with some trips being significantly longer or shorter than the mean.<br>
**Range** (20,260,150 seconds): The enormous range indicates that there are still some extremely long trips, suggesting the presence of outliers that greatly extend the duration spread.<br>


**Overall Insights:** <br>

Most bike trips are short and consistent, but there are a few extremely long trips causing a large spread in the data.
It would be valuable to investigate these outliers further to understand their nature and decide if they should be treated differently in the analysis.

```df['Trip Duration'].max()```


20260211


**Data Distribution Analysis** <br>

The goal of data distribution analysis is to understand how the values of a variable, like trip duration, are spread across the dataset. This helps to identify patterns, such as whether most trips are short or if there are many long trips, and to detect any extreme values (outliers) that could affect our analysis.

**Histogram** <br>

A histogram helps to see how frequently different values of trip duration occur. It shows whether the data is skewed (more data points on one side) or normally distributed.

```
import seaborn as sns
import matplotlib.pyplot as plt

# Plot a histogram of trip duration
sns.histplot(df['Trip Duration'], bins=30, kde=True)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```


![Distribution of trip duration](https://github.com/user-attachments/assets/5e11f92f-2be2-4cab-b76b-6f9d754f668f)


```
sns.histplot(df['Trip Duration'], bins=20, kde=True)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```

![Distribution of trip duration 20](https://github.com/user-attachments/assets/055e95f2-b8b1-4c0b-99a0-23a17947139b)



```
# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```


![Filtered distribution of trip durations](https://github.com/user-attachments/assets/78b11e0f-a033-4b09-ad4d-417f30a81c8b)


```
# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```



![Filtered distribution of trip durations2](https://github.com/user-attachments/assets/b54a4169-5c95-4cb0-8353-c69ebb380d3a)

The goal of using these plots is to better understand the distribution of trip durations by filtering out extreme values that may skew the visualization. In real-world data, outliers such as very long trips (e.g., over an hour or over 10,000 seconds) can distort the overall shape of the histogram and make it harder to observe typical patterns.

By applying thresholds like Trip Duration < 10,000 or Trip Duration < 3600, the plots focus on more reasonable, common trip durations. This makes the distribution easier to interpret and helps identify central tendencies and common trip lengths more clearly.



**Box Plot** <br>

A box plot helps to visualize the spread of the data and detect outliers. It shows the median (middle value), quartiles, and outliers.

```
sns.boxplot(x=df['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()
```


![Box Plot of Trip Durations](https://github.com/user-attachments/assets/42ef5a96-3704-4cac-b694-9a2da1460250)


```
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()
```


![Box Plot of Trip Durations 3600](https://github.com/user-attachments/assets/76d9f3c2-296e-48a9-80ea-a9fbf9d62e1d)


The box plot shows where most of the data points lie and highlights outliers that may need further investigation. The "box" represents the middle 50% of the data (interquartile range), and the "whiskers" extend to show the rest of the data, except for outliers.


```
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```


![Box Plot of Trip Durations Filtered 10000](https://github.com/user-attachments/assets/b707fc9a-145b-4bf2-a9d7-cef6f8869f25)


```
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 1000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```


![Box Plot of Trip Durations Filtered 1000](https://github.com/user-attachments/assets/5f18bcb4-2fb0-4875-a1ba-c2e98e6b068e)


Let's take a look at the values sorted by Trip Duration in descending order.

```
# Add a new column 'Trip Duration in Hours' by converting seconds to hours
df['Trip Duration(hrs)'] = df['Trip Duration'] / 3600

# Sort the DataFrame by 'Trip Duration' in descending order and select the top 30
top_50_max_duration = df.sort_values(by='Trip Duration', ascending=False).head(50)

# Select only the relevant columns, including the new 'Trip Duration in Hours'
top_50_max_duration = top_50_max_duration[['Trip Duration', 'Trip Duration(hrs)', 'Start Time', 'Stop Time']]

# Display the updated DataFrame
print(top_50_max_duration)
```

|        |   Trip Duration |   Trip Duration (hrs) | Start Time          | Stop Time           |
|-------:|----------------:|----------------------:|:--------------------|:--------------------|
| 374349 |        20260211 |             5627.84   | 2015-09-26 04:20:59 | 2016-05-17 16:11:10 |
| 109375 |        16329808 |             4536.06   | 2016-03-22 07:02:10 | 2016-09-27 07:05:38 |
|  78142 |         6065936 |             1684.98   | 2015-12-12 21:04:35 | 2016-02-21 02:03:32 |
|  71700 |         5366099 |             1490.58   | 2015-11-27 13:49:07 | 2016-01-28 16:24:07 |
| 313891 |         4826890 |             1340.8    | 2016-11-23 17:38:36 | 2017-01-18 14:26:46 |
|  95455 |         2104123 |              584.479  | 2016-02-12 07:27:56 | 2016-03-07 15:56:40 |
|  95491 |         2100551 |              583.486  | 2016-02-12 08:31:06 | 2016-03-07 16:00:18 |
|  95593 |         2071209 |              575.336  | 2016-02-12 16:32:54 | 2016-03-07 15:53:03 |
| 128729 |         1837255 |              510.349  | 2016-04-28 09:05:14 | 2016-05-19 15:26:09 |
|   8191 |         1620142 |              450.039  | 2015-10-13 23:52:48 | 2015-11-01 16:55:11 |
|  87178 |         1569765 |              436.046  | 2016-01-09 05:49:39 | 2016-01-27 09:52:25 |
| 243701 |         1532001 |              425.556  | 2016-09-11 16:32:21 | 2016-09-29 10:05:42 |
|   7135 |         1471896 |              408.86   | 2015-10-12 16:29:28 | 2015-10-29 17:21:05 |
| 226371 |         1258736 |              349.649  | 2016-08-26 23:19:01 | 2016-09-10 12:57:58 |
| 235239 |         1120971 |              311.381  | 2016-09-03 22:05:27 | 2016-09-16 21:28:18 |
| 266151 |         1021330 |              283.703  | 2016-10-01 15:01:46 | 2016-10-13 10:43:57 |
|   9932 |          942374 |              261.771  | 2015-10-16 09:59:46 | 2015-10-27 07:46:00 |
| 363996 |          871460 |              242.072  | 2017-03-09 08:59:22 | 2017-03-19 12:03:42 |
| 335320 |          802101 |              222.806  | 2017-01-10 08:49:55 | 2017-01-19 15:38:17 |
| 312564 |          721297 |              200.36   | 2016-11-21 23:29:17 | 2016-11-30 07:50:55 |
| 247265 |          623780 |              173.272  | 2016-09-14 12:54:47 | 2016-09-21 18:11:07 |
| 154321 |          622821 |              173.006  | 2016-06-07 13:20:54 | 2016-06-14 18:21:15 |
|   4282 |          496680 |              137.967  | 2015-10-08 14:49:02 | 2015-10-14 08:47:02 |
| 236155 |          488819 |              135.783  | 2016-09-04 21:11:32 | 2016-09-10 12:58:32 |
| 361199 |          399412 |              110.948  | 2017-03-03 19:08:11 | 2017-03-08 10:05:04 |
| 368395 |          390893 |              108.581  | 2017-03-24 19:45:28 | 2017-03-29 08:20:22 |
| 290143 |          361889 |              100.525  | 2016-10-25 10:34:43 | 2016-10-29 15:06:13 |
|   7807 |          357636 |               99.3433 | 2015-10-13 15:49:34 | 2015-10-17 19:10:11 |
| 328893 |          355757 |               98.8214 | 2016-12-22 13:00:20 | 2016-12-26 15:49:38 |
| 314068 |          353192 |               98.1089 | 2016-11-24 08:18:57 | 2016-11-28 10:25:29 |
| 111850 |          322385 |               89.5514 | 2016-03-27 01:10:51 | 2016-03-30 18:43:57 |
|  78139 |          310083 |               86.1342 | 2015-12-12 21:01:46 | 2015-12-16 11:09:50 |
|   1674 |          294734 |               81.8706 | 2015-10-05 07:42:52 | 2015-10-08 17:35:06 |
|  58136 |          264226 |               73.3961 | 2015-11-01 16:31:36 | 2015-11-04 17:55:23 |
|  96271 |          259489 |               72.0803 | 2016-02-17 10:33:53 | 2016-02-20 10:38:42 |
| 244959 |          251098 |               69.7494 | 2016-09-12 18:30:39 | 2016-09-15 16:15:38 |
| 263826 |          248376 |               68.9933 | 2016-09-28 18:17:15 | 2016-10-01 15:16:52 |
| 131542 |          237444 |               65.9567 | 2016-05-03 23:49:53 | 2016-05-06 17:47:17 |
| 105299 |          234181 |               65.0503 | 2016-03-12 13:11:30 | 2016-03-15 07:14:31 |
| 101557 |          234066 |               65.0183 | 2016-03-03 18:56:50 | 2016-03-06 11:57:57 |
|  89665 |          227014 |               63.0594 | 2016-01-16 17:16:13 | 2016-01-19 08:19:48 |
|  87985 |          222661 |               61.8503 | 2016-01-11 21:15:40 | 2016-01-14 11:06:42 |
| 364668 |          221604 |               61.5567 | 2017-03-10 17:39:21 | 2017-03-13 08:12:46 |
|  19260 |          203604 |               56.5567 | 2015-10-31 23:45:12 | 2015-11-03 07:18:37 |
|  73443 |          202474 |               56.2428 | 2015-12-02 18:45:13 | 2015-12-05 02:59:48 |
| 322619 |          199384 |               55.3844 | 2016-12-09 11:33:44 | 2016-12-11 18:56:48 |
| 351266 |          196255 |               54.5153 | 2017-02-16 07:45:17 | 2017-02-18 14:16:13 |
| 141839 |          194937 |               54.1492 | 2016-05-20 10:13:57 | 2016-05-22 16:22:54 |
| 274364 |          190956 |               53.0433 | 2016-10-10 02:13:37 | 2016-10-12 07:16:14 |
| 218222 |          188396 |               52.3322 | 2016-08-19 13:10:14 | 2016-08-21 17:30:11 |




**Unusually Long Durations:** The trip durations in the top 30 range from 496,680 seconds (about 5.7 days) to 20,260,211 seconds (about 234 days). For a bike-sharing service, this duration is extraordinarily long. Most bike trips typically last from a few minutes to a few hours, not multiple days.

**Suspicious Patterns:** The start and stop times for these trips span multiple days, weeks, or even months. This suggests that these trips could be due to data errors, such as trips that were not properly checked back in or recorded.

```
#2104123
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 200000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```



![Box Plot of Trip Durations Filtered Long Trips](https://github.com/user-attachments/assets/af5e2780-0182-4fe6-a72b-857f9072d4f9)


**How to get rid of outlers?** <br>

**The Quartile Method**, also known as the Interquartile Range (IQR) Method, is commonly used to identify and remove outliers from a dataset.

**Q1 (First Quartile)**: Imagine you have all your trip durations lined up from shortest to longest. Q1 is the point where 25% of the trips are shorter, and 75% are longer. It's like the mark where the first quarter of your data ends.

**Q3 (Third Quartile)**: Q3 is the point where 75% of the trips are shorter, and only 25% are longer. It's the mark where three-quarters of your data is below it, and only the top quarter is above.

**IQR (Interquartile Range)**: The IQR is the space between Q1 and Q3. It tells us how spread out the middle half of the trips are. So, if the IQR is small, most trips are of similar duration. If it’s big, the trip durations are more spread out.

Think of it like dividing your trip data into four equal parts: Q1 is the end of the first part, Q3 is the end of the third part, and the IQR is the chunk in between where the bulk of your data sits.

```
Q1 = df['Trip Duration'].quantile(0.25)
Q3 = df['Trip Duration'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:",IQR,"Q1:",Q1,"Q3:",Q3)
```


IQR: 409.0 Q1: 247.0 Q3: 656.0


**Q1 (25th percentile)**: 247 seconds (about 4.1 minutes)

**Q3 (75th percentile)**: 656 seconds (about 10.9 minute)

**IQR (Interquartile Range)**: 409 seconds (Q3 - Q1)

Using These Values to Identify Outliers: The IQR can be used to calculate the lower and upper bounds for detecting outliers.

**Calculate the Bounds:**

Lower Bound: Q1−1.5×IQR=247−1.5×409=247−613.5=−366.5 Since trip duration cannot be negative, the lower bound will be 0.

Upper Bound: Q3+1.5×IQR=656+1.5×409=656+613.5=1269.5

**Interpretation:**

Any trip duration below 0 seconds (which is impossible in this context) or above 1269.5 seconds (about 21.2 minutes) is considered an outlier. ered an outlier.


```
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Trip Duration'] < lower_bound) | (df['Trip Duration'] > upper_bound)]
print("outliers: trips longer than",upper_bound)
```

**outliers:** trips longer than 1269.5

```
#1269.5
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] <= 1269.5]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```


![Box Plot of Trip Durations Filtered 1269](https://github.com/user-attachments/assets/82382bfa-d349-4c65-8cda-66d7d0d96fba)



```df_filtered.shape```

(309029, 17)

```
median = df_filtered['Trip Duration'].median()
mean = df_filtered['Trip Duration'].mean()
mode = df_filtered['Trip Duration'].mode()[0]
std_trip = df_filtered['Trip Duration'].std()
range_trip = df_filtered['Trip Duration'].max() - df_filtered['Trip Duration'].min()
print("Median:",median,"Mean:",mean,"Mode:" , mode,"Std:", std_trip,"Range:", range_trip)
```

Median: 353.0 Mean: 428.31488306922654 Mode: 244 Std: 257.843141266675 Range: 1208


**Median is 353.0 seconds (about 5.88 minutes).**

The median represents the middle value of the trip durations, indicating that half of the bike trips are less than 353 seconds and the other half are more than 353 seconds. This suggests that many bike trips are fairly short.

**Mean (average) trip duration is 428.31 seconds.**

Since the mean is higher than the median, it suggests that there are some longer trips that are pulling the average up, though they are not as extreme as the original outliers before cleaning.

**Mode is 244 seconds (about 4.07 minutes).**

The mode of 244 seconds means that this is the most frequently occurring trip duration. Many bike trips are around 4 minutes, highlighting a common use pattern for short rides.

**Standard Deviation is 257.84 seconds.**

The standard deviation measures the spread of the trip durations. A value of 257.84 seconds indicates there is still a moderate amount of variation, but it is more reasonable than before outlier removal. Most trips fall within a range of about 4 minutes from the mean.

**Range is 1208 seconds (about 20.13 minutes).**

The range is 1208 seconds, showing the difference between the longest and shortest trips in the filtered dataset. This confirms that, even after cleaning, there is still some variation in trip durations, but the data now reflects a more typical and expected usage pattern.

**Insights:** <br>

The median and mode indicate that most bike trips are short, commonly around 4 to 6 minutes. The mean being slightly higher than the median suggests that there are still a few longer trips influencing the average, but they are not excessively skewing the data. The cleaned dataset now gives a clearer understanding of bike-sharing behavior, showing that the service is mostly used for short trips, with occasional longer ones.nal longer ones.

```
# Example: Filtering out trip durations longer than a reasonable threshold
#df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```

![Filtered Distribution of Trip Durations 3600](https://github.com/user-attachments/assets/8cac5547-a62f-41a6-ad7c-e42f805b23c3)


This histogram represents the filtered distribution of trip durations. Here's how to interpret it:

**Shape of the Distribution:** <br>

The distribution is right-skewed, meaning that most bike trips are short in duration, and as the duration increases, the number of trips decreases. The peak of the histogram occurs around 200-300 seconds (3.3 to 5 minutes), indicating that most bike trips are concentrated around this duration.

**Frequency:** <br>

The y-axis shows the frequency of trip durations, with the highest frequency reaching around 30,000 trips. This indicates that a large number of bike trips are in the shorter duration range. As the duration increases beyond 300 seconds, the frequency steadily decreases, showing that longer trips are less common.

**Tail of the Distribution:** <br>

The right tail extends up to around 1200 seconds (20 minutes), with relatively fewer trips occurring in this range. This confirms that while some longer trips still occur, they are much less frequent compared to the shorter ones.

**Insights:** <br>

The majority of bike trips are short, typically between 3 to 7 minutes, which is consistent with the nature of a bike-sharing service meant for quick rides. The right-skewed nature of the data suggests that there are still some longer trips, but they are not as extreme as the original outliers. This distribution helps us understand typical usage patterns and indicates that most users prefer short-duration rides. ation rides.

```
df_filtered.shape
df = df_filtered
```

```
# Count the total number of records and the number of 'Subscriber' user types
total_users = df.shape[0]
subscriber_count = df[df['User Type'] == 'Subscriber'].shape[0]
customer_count = df[df['User Type'] == 'Customer'].shape[0]

# Calculate the percentage of 'Subscriber' user types
subscriber_percentage = round((subscriber_count / total_users) * 100,2)
customer_percentage = round((customer_count / total_users) * 100,2)
print("Percentage of Subscribers:", subscriber_percentage, "%, while just", customer_percentage,"% is not subscribed")
```


Percentage of Subscribers: 96.06 %, while just 3.94 % is not subscribed.


**How does trip number vary on user type?**

```
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")
```

```
values = df['User Type'].value_counts()
labels = df['User Type'].value_counts().index

#Plot the pie chart
plt.pie(values, labels=labels, autopct = '%1.1f%%', startangle=140)

#Add a title
plt.title("User Type variation")

#Show the chart
plt.axis('equal')
plt.show()
```


![User Type Variation](https://github.com/user-attachments/assets/c9a47afc-759b-4da3-a55a-47c4b4399508)



**How does trip number vary based on gender?** <br>

I started with the data cleaned from outliers.

```
gender_counts = df['Gender'].value_counts()

#Extract values and labels for the pie chart
values = gender_counts.values
labels = gender_counts.index

#Create a pie chart
plt.figure(figsize=(8, 6))
plt.pie(
    values,
    labels=labels,
    autopct='%1.1f%%',
    startangle=0,
    shadow=False
)

#Add a title
plt.title("Gender Variation in Trip Numbers")

#Display the chart
plt.show()
```


![Gender Variation in Trip Numbers](https://github.com/user-attachments/assets/6b4ebda0-2cf2-4acc-b84d-ca8fdcb7425b)



```
# Group the data by 'Gender' and calculate summary statistics for 'Trip Duration'
gender_summary = df.groupby('Gender')['Trip Duration'].agg(['mean', 'median', 'count'])

# Rename the columns for clarity
gender_summary.rename(columns={'mean': 'Average Trip Duration', 'median': 'Median Trip Duration', 'count': 'Number of Trips'}, inplace=True)

# Display the summary statistics
print(gender_summary)
```

**Trip Duration Statistics and Total Number of Trips by Gender**

| Gender | Average Trip Duration (sec) | Mediam Trip Duration (sec) | Number of Trips |
|--------|-----------------------------|----------------------------|-----------------|
|   0    |          610.04             |            562.0           |     17,550      |
|   1    |          412.17             |            336.0           |    227,201      |
|   2    |          435.77             |            372.0           |     64,278      |


```
# Calculate average and median trip durations by gender
gender_trip_duration = df_filtered.groupby('Gender')['Trip Duration'].agg(['mean', 'median'])
print(gender_trip_duration)

# Visualize with a box plot
sns.boxplot(x='Gender', y='Trip Duration', data=df_filtered)
plt.title('Trip Duration by Gender')
plt.xlabel('Gender (0 = Not Specified, 1 = Male, 2 = Female)')
plt.ylabel('Trip Duration (seconds)')
plt.show()
```

**Average and Median Trip Duration by Gender**

| Gender | Average Trip Duration (sec) | Mediam Trip Duration (sec) | 
|--------|-----------------------------|----------------------------|
|   0    |          610.04             |            562.0           |
|   1    |          412.17             |            336.0           |
|   2    |          435.77             |            372.0           |



![Trip Duration by Gender](https://github.com/user-attachments/assets/4210f8ad-950c-4f9e-8165-1b01848a01c6)


```
# Count the number of trips for each gender
trip_count_by_gender = df_filtered['Gender'].value_counts()
print(trip_count_by_gender)

# Visualize with a bar plot
sns.barplot(x=trip_count_by_gender.index, y=trip_count_by_gender.values)
plt.title('Trip Frequency by Gender')
plt.xlabel('Gender (0 = Not Specified, 1 = Male, 2 = Female)')
plt.ylabel('Number of Trips')
plt.show()
```


**Number of Trips by Gender**

| Gender | Number of Trips | 
|--------|-----------------|
|   1    |    227,201      | 
|   2    |     64,278      | 
|   0    |     17,550      |


![Trip Frequency by Gender](https://github.com/user-attachments/assets/ec91313b-b2c8-45d4-b2c9-702f0a0b50b3)


The analysis of NYC Bike Share usage shows significant differences in how often and how long people ride, depending on their gender:

Male users (Gender = 1) take the most trips by far — over 227,000 rides.
Female users (Gender = 2) are the second-largest group, with about 64,000 rides.
Users who didn’t specify their gender (Gender = 0) take the fewest trips, around 17,500.

Interestingly, even though users with unspecified gender take the least number of trips, their average and median trip durations are the longest. In contrast, male riders have the shortest trips on average, even though they ride most frequently. Female users fall in between — their trip durations are longer than male users but shorter than those with unspecified gender.

This suggests that gender may not only influence how frequently people use the service but also how long they ride when they do.


```
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Adding a column 'Age' based on 'Birth Year'
df['Age'] = 2017 - df['Birth Year']
df['User Type Encoded'] = df['User Type'].map({'Subscriber': 1, 'Customer': 0})

print(df.head())

#Correlation matrix for numeric fields
correlation_matrix = df[['Trip_Duration_in_min', 'Age', 'Start Station Latitude', 'Start Station Longitude', 'End Station Latitude', 'End Station Longitude',
                        'Gender', 'User Type Encoded']].corr()

#Let's build a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix for Trip Duration and Other Numeric Variables")
plt.show()
```

**Correlation Matrix**


| Trip Duration | Start Time          | Stop Time           | Start Station Name | End Station Name     | Bike ID | User Type   | Birth Year | Gender | Age |
|---------------|---------------------|----------------------|--------------------|----------------------|---------|-------------|------------|--------|-----|
| 376           | 2015-10-01 00:16:26 | 2015-10-01 00:22:42  | Christ Hospital    | Oakland Ave          | 24470   | Subscriber  | 1960.0     | 1      | 57  |
| 739           | 2015-10-01 00:27:12 | 2015-10-01 00:39:32  | Oakland Ave        | Christ Hospital      | 24481   | Subscriber  | 1960.0     | 1      | 57  |
| 275           | 2015-10-01 00:34:31 | 2015-10-01 00:39:06  | Newport Pkwy       | Warren St            | 24613   | Subscriber  | 1975.0     | 1      | 42  |
| 561           | 2015-10-01 00:40:12 | 2015-10-01 00:49:33  | Exchange Place     | Liberty Light Rail   | 24668   | Customer    | 1984.0     | 0      | 33  |
| 365           | 2015-10-01 00:41:46 | 2015-10-01 00:47:51  | Heights Elevator   | Central Ave          | 24644   | Customer    | 1984.0     | 0      | 33  |


![Correlation Matrix for Trip Duration](https://github.com/user-attachments/assets/8fe82549-b9f7-4fb4-b336-ff61839a0b69)


This heatmap displays the correlation matrix for "Trip Duration" and other numeric variables. It visually shows how strongly and in which direction (positive or negative) each pair of variables is correlated.

**Diagonal Values:**

All diagonal values are 1 because a variable is perfectly correlated with itself.

**Trip Duration vs. Other variables:**

Start Station Latitude r= 0.15: a weak positive correlation. End Station Latitude r= 0.11: a weak positive correlation. User Type Encoded r= – 0.21: a weak negative correlation. Other variables (Age, Longitudes, Gender) show very weak correlations close to 0.

**Start Station Latitude and End Station Latitude:**

Correlation of 0.48 – a moderate positive correlation.

**Start Station Longitude and End Station Longitude:**

Correlation of 0.57 – a moderate to strong positive correlation, suggesting that longitudes are often similar for start and end stations.

**Gender and User Type Encoded:**

Correlation of 0.47 – a moderate positive correlation between gender and user type (subscriber and customer).


```
import seaborn as sns
import matplotlib.pyplot as NotImplemented

#Cross-tabulation of User Type and Gender
cross_tab = pd.crosstab(df['Gender'], df['User Type'])
print(cross_tab)

#Plot the relationship
sns.countplot(x='Gender', hue='User Type', data=df)
plt.title("User Type Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="User Type")
plt.show()
```

**User Type Distribution by Gender**

| **User Type** | **Customer** | **Subscriber** |
|---------------|--------------|----------------|
| **Gender 0**  | 12142        | 5408           |
| **Gender 1**  | 18           | 227183         |
| **Gender 2**  | 4            | 64274          |




![User Type Distribution by Gender](https://github.com/user-attachments/assets/99514afb-a286-4f35-a819-bea0dc49c10a)


The moderate positive correlation between gender and user type (subscriber and customer) of 0.47 makes sense because:

Males (1) and Females (2) are predominantly Subscribers (User Type = 1).
Unknown (0) users have a relatively high proportion in the Customer (User Type=0) category.


```
#Cross-tabulation for stacked bar plot
cross_tab = pd.crosstab(df['Gender'], df['User Type'])

#Normalize for percentages
cross_tab_percentage = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

#Plot a stacked bar chart
cross_tab_percentage.plot(kind='bar', stacked=True, color=['coral', 'lightblue'])
plt.title("User Type Distribution by Gender")
plt.ylabel("Percentage")
plt.xlabel("Gender")
plt.legend(title="User Type", labels=["Customer", "Subscriber"])
plt.show()
```



![User Type Distribution by Gender %](https://github.com/user-attachments/assets/a27e924b-3a5b-40ca-96ae-51117328eecc)


```
#Add a new column to check if start and end stations are the same
df['Same Station'] = df['Start Station ID'] == df['End Station ID']

#Calculate the percentage of trips where the start and end stations are the same
same_station_percentage = df['Same Station'].mean() * 100

#Print the result
print(f"Percentage of trips with the same start and end stations: {same_station_percentage:.2f}%")
```

Percentage of trips with the same start and end stations: 2.42%


```
df['Start Time'] = pd.to_datetime(df['Start Time'])

#Extract numerical features from Start Time
df['Start Hour'] = df['Start Time'].dt.hour       #Hour of the day
df['Start Day'] = df['Start Time'].dt.dayofweek   #Day of the week (0=Monday, 6=Sunday)
df['Start Month'] = df['Start Time'].dt.month     #Month of the year

#Recalculate the correlation matrix
correlation_matrix = df[['Trip_Duration_in_min', 'Age', 'Start Station Latitude',
                        'Start Station Longitude', 'End Station Latitude',
                        'End Station Longitude', 'Start Hour', 'Start Day', 'Start Month']].corr()

#Visualize the updated correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix Including Start Time Features")
plt.show()
```



![Correlation Matrix Including Start Time Features](https://github.com/user-attachments/assets/dba8677b-b381-4ccc-b063-64fadebb2a66)




**Trip Duration vs Other Variables:**

Start Station Latitude r=0.15 – a weak positive correlation. End Station Latitude r=0.11 – a weak positive correlation. User Type Encoded r=-0.21 – a weak negative correlation.


```
import seaborn as sns
import matplotlib.pyplot as plt

#Group by hour and calculate average trip duration
avg_duration_hour = df.groupby('Start Hour')['Trip_Duration_in_min'].mean()

#Plot the results
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_duration_hour.index, y=avg_duration_hour.values, marker='o')
plt.title("Average Trip Duration by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Trip Duration (minutes)")
plt.show()
```



![Average Trip duration by Hour of Day](https://github.com/user-attachments/assets/c91ee70b-246f-4aa2-a6d6-bef6cd6a25b2)



To have actual days of a week in x axis, I created a mapping.


```
df_filtered['Start Time'] = pd.to_datetime(df['Start Time'])
df_filtered['Start Day'] = df['Start Time'].dt.dayofweek     #0 = Monday, 6 = Sunday

#Map numeric days to day names
day_mapping = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday"
}
df['Day Name'] = df['Start Day'].map(day_mapping)

#Group by day name and calculate average trip duration
avg_duration_day = df.groupby('Day Name')['Trip_Duration_in_min'].mean()

#Sort days in weekday order
avg_duration_day = avg_duration_day.reindex(["Monday", "Tuesday", "Wednesday", 
                                            "Thursday", "Friday", "Saturday", "Sunday"])

#Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_duration_day.index, y=avg_duration_day.values, palette="winter")
plt.title("Average Trip Duration by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Average Trip Duration (minutes)")
plt.show()
```



![Average Trip duration by Day of the Week](https://github.com/user-attachments/assets/500f2dc4-56b4-4080-b6cc-e223b0c61b09)



**Conclusion:**

This project highlights the effectiveness of Python for Exploratory Data Analysis (EDA). Using libraries like Pandas, Seaborn, Matplotlib, and Plotly, I cleaned, explored, and visualized the data efficiently, revealing meaningful patterns and insights.

Throughout the analysis, I applied techniques such as data cleaning, grouping, aggregation, and visualization to better understand trends in bike trip data. Functions like groupby() and agg(), along with various plots, allowed to examine differences in trip duration and frequency by gender, making the data easier to interpret and draw conclusions from.

**Key Takeaways:**

Quick Insights: Python makes it easy to dive into a dataset and helps spotting patterns and key metrics right away.

Flexibility: With so many great libraries available, Python gives you the freedom to handle everything from basic summaries to more complex analysis without much hassle.

Clear Visualization: Libraries like Seaborn and Matplotlib made it simple to turn raw data into clear, compelling visuals that help tell the story behind the numbers.

Overall, Python proved to be a reliable tool for exploratory data analysis. It gave everything to clean, explore, and visualize the data—making it easier to understand what’s going on and support data-informed decisions.
