import pandas as pd
file_path = "usa_monthly_temperature_1900_2023.csv"  # Replace with your file path
data = pd.read_csv(file_path)
print("Missing values before cleaning:")
print(data.isnull().sum())
data["Avg_Temperature_Celsius"] = pd.to_numeric(data["Avg_Temperature_Celsius"], errors='coerce')
data["Avg_Temperature_Celsius"] = data["Avg_Temperature_Celsius"].interpolate(method='linear')
data_cleaned = data.drop_duplicates()
data_cleaned["Date"] = pd.to_datetime(data_cleaned[["Year", "Month"]].assign(DAY=1))
data_cleaned = data_cleaned[["Date", "Year", "Month", "Avg_Temperature_Celsius"]]
output_file_path = "us_monthly_temperature_cleaned_1900_2023.csv" 
data_cleaned.to_excel(output_file_path, index=False)
print(f"Cleaned data saved to: {output_file_path}")
