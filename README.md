# New York City Bike Share EDA

In this project, I conducted Exploratory Data Analysis (EDA) on the New York City Bike Share dataset using Python. 
[This dataset](https://www.kaggle.com/datasets/akkithetechie/new-york-city-bike-share-dataset) can be found on Kaggle.
My analysis focused on understanding trip duration patterns, user demographics, and data quality issues. 
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

# Exploring the Dataset

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

Finally, I use df.head() to display the first 5 rows of the DataFrame. This gives a quick snapshot of the data, so I can see what it looks like and start 
understanding the structure of the dataset, such as column names and sample values.

```df.head(5)```

TABLE



The command df.info() is used to quickly get an overview of the dataset, including:

The number of rows and columns.
The data types of each column (e.g., integer, float, object).
The number of non-null values in each column, which helps identify missing data. This method provides a quick summary of the dataset's structure and allows 
to spot potential issues, such as columns with missing values or incorrect data types.


```df.info()```

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 735502 entries, 0 to 735501
Data columns (total 16 columns):

![df info](https://github.com/user-attachments/assets/b5596ce8-e111-41f9-91ca-6b8b9c763253)







# Let's start EDA.

```df.shape```

(735502, 16)
There are 16 columns and 735502 rows in the dataset.

Let's first check if there are any duplicates. The first sep - is to clear our dataset from duplicates.

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
Median (384 seconds): The median trip duration is around 6.4 minutes, indicating that half of the bike trips are shorter than this time, showing that most trips are relatively brief.<br>
Mode (244 seconds): The most common trip duration is 244 seconds (about 4 minutes), reflecting a large number of quick bike trips.<br>
Standard Deviation (48,685.70 seconds): The data shows a high variability in trip durations, with some trips being significantly longer or shorter than the mean.<br>
Range (20,260,150 seconds): The enormous range indicates that there are still some extremely long trips, suggesting the presence of outliers that greatly extend the duration spread.<br>

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

GRAPH


```
# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```


GRAPH

```
# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()
```


GRAPH


**Box Plot** <br>
A box plot helps to visualize the spread of the data and detect outliers. It shows the median (middle value), quartiles, and outliers.

```
sns.boxplot(x=df['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()
```

GRAPH


```
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()
```

GRAPH


The box plot shows where most of the data points lie and highlights outliers that may need further investigation. The "box" represents the middle 50% of the data (interquartile range), and the "whiskers" extend to show the rest of the data, except for outliers.


```
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```

GRAPH

```
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] < 1000]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```

GRAPH


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


TABLE








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


GRAPH


***How to get rid of outlers?*** <br>

The Quartile Method, also known as the Interquartile Range (IQR) Method, is commonly used to identify and remove outliers from a dataset.

Q1 (First Quartile): Imagine you have all your trip durations lined up from shortest to longest. Q1 is the point where 25% of the trips are shorter, and 75% are longer. It's like the mark where the first quarter of your data ends.

Q3 (Third Quartile): Q3 is the point where 75% of the trips are shorter, and only 25% are longer. It's the mark where three-quarters of your data is below it, and only the top quarter is above.

IQR (Interquartile Range): The IQR is the space between Q1 and Q3. It tells us how spread out the middle half of the trips are. So, if the IQR is small, most trips are of similar duration. If it’s big, the trip durations are more spread out.

Think of it like dtheing your trip data into four equal parts: Q1 is the end of the first part, Q3 is the end of the third part, and the IQR is the chunk in between where the bulk of your data sits.

```
Q1 = df['Trip Duration'].quantile(0.25)
Q3 = df['Trip Duration'].quantile(0.75)
IQR = Q3 - Q1
print("IQR:",IQR,"Q1:",Q1,"Q3:",Q3)
```


IQR: 409.0 Q1: 247.0 Q3: 656.0


Q1 (25th percentile): 247 seconds (about 4.1 minutes)

Q3 (75th percentile): 656 seconds (about 10.9 minute)

IQR (Interquartile Range): 409 seconds (Q3 - Q1)

Using These Values to Identify Outliers: The IQR can be used to calculate the lower and upper bounds for detecting outliers.

Calculate the Bounds:

Lower Bound: Q1−1.5×IQR=247−1.5×409=247−613.5=−366.5 Since trip duration cannot be negative, the lower bound will be 0.

Upper Bound: Q3+1.5×IQR=656+1.5×409=656+613.5=1269.5

Interpretation:

Any trip duration below 0 seconds (which is impossible in this context) or above 1269.5 seconds (about 21.2 minutes) is considered an outlier. ered an outlier.


```
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['Trip Duration'] < lower_bound) | (df['Trip Duration'] > upper_bound)]
print("outliers: trips longer than",upper_bound)
```

outliers: trips longer than 1269.5

```
#1269.5
# Example: Filtering out extremely long trips
df_filtered = df[df['Trip Duration'] <= 1269.5]  # Adjust the threshold as needed
sns.boxplot(x=df_filtered['Trip Duration'])
plt.title('Box Plot of Trip Durations (Filtered)')
plt.show()
```


GRAPH

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
Median is 353.0 seconds (about 5.88 minutes).

The median represents the middle value of the trip durations, indicating that half of the bike trips are less than 353 seconds and the other half are more than 353 seconds. This suggests that many bike trips are fairly short.

Mean (average) trip duration is 428.31 seconds.

Since the mean is higher than the median, it suggests that there are some longer trips that are pulling the average up, though they are not as extreme as the original outliers before cleaning.

Mode is 244 seconds (about 4.07 minutes).

The mode of 244 seconds means that this is the most frequently occurring trip duration. Many bike trips are around 4 minutes, highlighting a common use pattern for short rides.

Standard Deviation is 257.84 seconds.

The standard deviation measures the spread of the trip durations. A value of 257.84 seconds indicates there is still a moderate amount of variation, but it is more reasonable than before outlier removal. Most trips fall within a range of about 4 minutes from the mean.

Range is 1208 seconds (about 20.13 minutes).

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

GRAPH


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



PIE CHART

**How does trip number vary based on gender?** <br>

We are starting with the data cleaned from outliers.

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


PIE CHART



```
# Group the data by 'Gender' and calculate summary statistics for 'Trip Duration'
gender_summary = df.groupby('Gender')['Trip Duration'].agg(['mean', 'median', 'count'])

# Rename the columns for clarity
gender_summary.rename(columns={'mean': 'Average Trip Duration', 'median': 'Median Trip Duration', 'count': 'Number of Trips'}, inplace=True)

# Display the summary statistics
print(gender_summary)
```



