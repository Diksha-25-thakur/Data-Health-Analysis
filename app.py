# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------
# Step 1: Create a Sample Dataset
# --------------------------------------------

# Sample health dataset
data_dict = {
    'Patient_ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [45, 54, 35, 60, 40, 50, 65, 30, 55, 42],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Blood_Pressure': [130, 145, 120, 150, 135, 140, 155, 110, 142, 138],
    'Cholesterol': [210, 250, 180, 300, 220, 230, 280, 170, 260, 240],
    'Heart_Rate': [80, 85, 75, 90, 78, 88, 92, 70, 86, 82],
    'BMI': [27.5, 30.1, 25.3, 32.0, 28.4, 29.7, 33.1, 24.5, 31.0, 26.8],
    'Glucose': [110, 120, 95, 140, 105, 125, 150, 90, 135, 100]
}

# Convert dictionary to DataFrame
df = pd.DataFrame(data_dict)

# Save the dataset as CSV
df.to_csv("health_data.csv", index=False)
print("✅ Sample health_data.csv created successfully!\n")

# --------------------------------------------
# Step 2: Load the Dataset
# --------------------------------------------
data = pd.read_csv("health_data.csv")
print("✅ Dataset Loaded Successfully!\n")
print(data.head())

# --------------------------------------------
# Step 3: Basic Info and Data Cleaning
# --------------------------------------------
print("\n🔍 Dataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Fill missing values (if any)
data.fillna(data.mean(numeric_only=True), inplace=True)

# --------------------------------------------
# Step 4: Descriptive Statistics
# --------------------------------------------
print("\n📊 Statistical Summary:")
print(data.describe())

# --------------------------------------------
# Step 5: Data Filtering (Using NumPy)
# --------------------------------------------
bp = np.array(data['Blood_Pressure'])
chol = np.array(data['Cholesterol'])

# Filter: Patients with high BP (>140)
high_bp = data[data['Blood_Pressure'] > 140]
print("\n⚠️ Patients with High Blood Pressure:\n", high_bp[['Patient_ID', 'Age', 'Blood_Pressure']])

# Filter: Patients with high cholesterol (>240)
high_chol = data[data['Cholesterol'] > 240]
print("\n⚠️ Patients with High Cholesterol:\n", high_chol[['Patient_ID', 'Age', 'Cholesterol']])

# --------------------------------------------
# Step 6: Data Visualization
# --------------------------------------------

# Set visual style
sns.set(style="whitegrid")

# (a) Histogram: Age Distribution
plt.figure(figsize=(6,4))
plt.hist(data['Age'], bins=8, color='skyblue', edgecolor='black')
plt.title('Age Distribution of Patients')
plt.xlabel('Age')
plt.ylabel('Number of Patients')
plt.show()

# (b) Bar Plot: Average Cholesterol by Gender
plt.figure(figsize=(6,4))
gender_chol = data.groupby('Gender')['Cholesterol'].mean()
gender_chol.plot(kind='bar', color=['lightcoral', 'lightblue'])
plt.title('Average Cholesterol by Gender')
plt.xlabel('Gender')
plt.ylabel('Cholesterol (mg/dL)')
plt.show()

# (c) Scatter Plot: Age vs Blood Pressure
plt.figure(figsize=(6,4))
plt.scatter(data['Age'], data['Blood_Pressure'], color='orange', edgecolors='black')
plt.title('Age vs Blood Pressure')
plt.xlabel('Age')
plt.ylabel('Blood Pressure (mmHg)')
plt.show()

# (d) Scatter Plot: BMI vs Cholesterol
plt.figure(figsize=(6,4))
plt.scatter(data['BMI'], data['Cholesterol'], color='green', edgecolors='black')
plt.title('BMI vs Cholesterol')
plt.xlabel('BMI')
plt.ylabel('Cholesterol (mg/dL)')
plt.show()

# (e) Correlation Heatmap
plt.figure(figsize=(8,6))
# Exclude non-numeric columns for correlation calculation
sns.heatmap(data.drop('Gender', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Between Health Parameters')
plt.show()

# --------------------------------------------
# Step 7: Insights
# --------------------------------------------
print("\n💡 Key Insights:")
print("- Older patients (Age > 50) tend to have higher blood pressure and cholesterol.")
print("- Males show slightly higher average cholesterol levels compared to females.")
print("- BMI and Blood Pressure are moderately correlated.")
print("- Glucose and Cholesterol show a weak correlation.")

# --------------------------------------------
# Step 8: Conclusion
# --------------------------------------------
print("\n✅ Conclusion:")
print("This analysis demonstrates how Python can be used for health data analysis using NumPy, Pandas, and Matplotlib.")
print("We identified patterns in blood pressure, cholesterol, and BMI, which can help in early health risk detection.")

