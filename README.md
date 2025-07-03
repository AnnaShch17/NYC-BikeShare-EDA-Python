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

#Exploring the Dataset

First, let's load data into a DataFrame using pd.read_csv(). This function reads the data from the CSV file located in the specified path and stores it in the variable df. 
Now, df holds all the bike-share data from New York City, collected between 2015 and 2017.

```df= pd.read_csv("../input/new-york-city-bike-share-dataset/NYC-BikeShare-2015-2017-combined.csv")```

Next, I removed a column named 'Unnamed: 0'. This column is likely an unnecessary index column created during data export, so I do not need it. 
I used df.drop() to remove it, specifying axis=1 to drop a column (not a row) and inplace=True to make the change directly to df.

```df.drop("Unnamed: 0",axis=1,inplace =True)```

Finally, I use df.head() to display the first 5 rows of the DataFrame. This gives a quick snapshot of the data, so I can see what it looks like and start 
understanding the structure of the dataset, such as column names and sample values.

```df.head(5)```



The command df.info() is used to quickly get an overview of the dataset, including:

The number of rows and columns.
The data types of each column (e.g., integer, float, object).
The number of non-null values in each column, which helps identify missing data. This method provides a quick summary of the dataset's structure and allows 
to spot potential issues, such as columns with missing values or incorrect data types.



#Let's start EDA.

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

Descriptive Statistics<br>
Mean: The average value of a dataset.<br>
Median: The middle value when the data is sorted.<br>
Mode: The most frequently occurring value.<br>
Standard Deviation: Measures how spread out the data is from the mean.<br>
Range: The difference between the maximum and minimum values.

```mean_duration = df['Trip Duration'].mean()
median_duration = df['Trip Duration'].median()
mode_duration = df['Trip Duration'].mode()[0]
std_dev_duration = df['Trip Duration'].std()  # Standard Deviation
range_durtion = df['Trip Duration'].max() - df['Trip Duration'].min()  # Range
print("Mean:", mean_duration, "Median:", median_duration, "Mode:", mode_duration,"Std Dev:", std_dev_duration, "Range :", range_durtion)```



Mean: 962.9629203227137 Median: 384.0 Mode: 244 Std Dev: 48685.70128924114 Range : 20260150
Based on the descriptive statistics of the Trip Duration data:<br>
Mean (962.96 seconds): The average trip duration is about 16 minutes, suggesting that, on average, bike trips are relatively short.<br>
Median (384 seconds): The median trip duration is around 6.4 minutes, indicating that half of the bike trips are shorter than this time, showing that most trips
are relatively brief.<br>
Mode (244 seconds): The most common trip duration is 244 seconds (about 4 minutes), reflecting a large number of quick bike trips.<br>
Standard Deviation (48,685.70 seconds): The data shows a high variability in trip durations, with some trips being significantly longer or shorter than the mean.<br>
Range (20,260,150 seconds): The enormous range indicates that there are still some extremely long trips, suggesting the presence of outliers that greatly extend
the duration spread.


##Overall Insights:

Most bike trips are short and consistent, but there are a few extremely long trips causing a large spread in the data.
It would be valuable to investigate these outliers further to understand their nature and decide if they should be treated differently in the analysis.

```df['Trip Duration'].max()```

20260211
Data Distribution Analysis
The goal of data distribution analysis is to understand how the values of a variable, like trip duration, are spread across the dataset. This helps to identify patterns,
such as whether most trips are short or if there are many long trips, and to detect any extreme values (outliers) that could affect our analysis.

Histogram
A histogram helps to see how frequently different values of trip duration occur. It shows whether the data is skewed (more data points on one side) or normally distributed.

```import seaborn as sns
import matplotlib.pyplot as plt

# Plot a histogram of trip duration
sns.histplot(df['Trip Duration'], bins=30, kde=True)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()```






```sns.histplot(df['Trip Duration'], bins=20, kde=True)
plt.title('Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()```



```# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 10000]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()```

graph

```# Example: Filtering out trip durations longer than a reasonable threshold
df_filtered = df[df['Trip Duration'] < 3600]  # Adjust the threshold as needed
sns.histplot(df_filtered['Trip Duration'], bins=30, kde=True)
plt.title('Filtered Distribution of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.ylabel('Frequency')
plt.show()```

graph




Box Plot
A box plot helps to visualize the spread of the data and detect outliers. It shows the median (middle value), quartiles, and outliers.


```sns.boxplot(x=df['Trip Duration'])
plt.title('Box Plot of Trip Durations')
plt.xlabel('Duration (seconds)')
plt.show()```
