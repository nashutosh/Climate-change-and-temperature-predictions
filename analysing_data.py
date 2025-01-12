import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('us_monthly_temperature_cleaned_1900_2023.csv')

# First plot: Temperature Distribution
plt.figure(figsize=(10, 8))
sns.histplot(data['Avg_Temperature_Celsius'], bins=30, kde=True)
plt.title('Temperature Distribution')
plt.xlabel('Temperature (°C)')
plt.show()

# Second plot: Temperature Trend (1900-2023)
yearly_avg = data.groupby('Year')['Avg_Temperature_Celsius'].mean().reset_index()
plt.figure(figsize=(10, 8))
sns.regplot(data=yearly_avg, x='Year', y='Avg_Temperature_Celsius', scatter_kws={'alpha':0.5})
plt.title('Temperature Trend (1900-2023)')
plt.show()

# Third plot: Monthly Temperature Patterns
monthly_avg = data.groupby('Month')['Avg_Temperature_Celsius'].mean().reset_index()
plt.figure(figsize=(10, 8))
sns.barplot(data=monthly_avg, x='Month', y='Avg_Temperature_Celsius')
plt.title('Monthly Temperature Patterns')
plt.show()

# Fourth plot: State-Month Temperature Heatmap
pivot_data = data.pivot_table(values='Avg_Temperature_Celsius', index='State', columns='Month', aggfunc='mean')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_data, cmap='RdYlBu_r', center=0)
plt.title('State-Month Temperature Heatmap')
plt.show()

# Key Statistics
print('Key Statistics:')
print('Average Temperature:', round(data['Avg_Temperature_Celsius'].mean(), 2), '°C')
print('Temperature Range:', round(data['Avg_Temperature_Celsius'].min(), 2), 'to', 
      round(data['Avg_Temperature_Celsius'].max(), 2), '°C')
print('States:', len(data['State'].unique()))
