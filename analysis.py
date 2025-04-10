import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('Traffic.csv')

# Fix datetime formats
df['Time'] = pd.to_datetime(df['Time'], format='%I:%M:%S %p')  # e.g., '01:23:45 PM'
df['Hour'] = df['Time'].dt.hour

# Convert full date
df['Date'] = pd.to_datetime(df['Date'])  # e.g., '2024-04-01'

# Add day of the week
df['Day of the week'] = df['Date'].dt.day_name()

# Define vehicle count columns
vehicle_cols = ['CarCount', 'BikeCount', 'BusCount', 'TruckCount']

# Handle missing values (optional - you can choose dropna or fillna)
df[vehicle_cols] = df[vehicle_cols].fillna(0)

# Calculate total vehicles
df['Total'] = df[vehicle_cols].sum(axis=1)

# Check for any missing values
print("Missing values:\n", df.isnull().sum())

# 1. Bar plot of total vehicles by type
df[vehicle_cols].sum().plot(kind='bar', figsize=(8, 5), color='orange')
plt.title("Total Vehicles by Type")
plt.ylabel("Count")
plt.grid()
plt.show()

# 2. Line plot of total traffic by hour
plt.figure(figsize=(10, 5))
sns.lineplot(x='Hour', y='Total', data=df, ci=None)
plt.title('Traffic Volume by Hour')
plt.xlabel('Hour of Day')
plt.ylabel('Vehicle Count')
plt.grid()
plt.show()

# 3. Box plot of traffic by day of week
plt.figure(figsize=(8, 4))
sns.boxplot(x='Day of the week', y='Total', data=df)
plt.title("Traffic Distribution by Day")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# 4. Correlation heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(df[vehicle_cols + ['Total']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between Vehicle Types and Total")
plt.show()
