import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('lsml.csv', sep=',')

# Display the first few rows
print(df.head())

# Display basic information
print(df.info())

# Display summary statistics
print(df.describe())

# # Display distribution of the target variable
# plt.figure(figsize=(10, 6))
# sns.histplot(df['median_income'], kde=True)
# plt.title('Distribution of Median Income')
# plt.xlabel('Median Income')
# plt.ylabel('Frequency')
# #plt.show()

# # Display correlation matrix
# plt.figure(figsize=(12, 10))
# sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
# plt.title('Correlation Matrix')
# #plt.show()

# Select features and target
X = df[['median_income']]
y = df['median_house_value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R2 Score:', r2_score(y_test, y_pred))

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.title('Linear Regression')
plt.xlabel('Median Income')
plt.ylabel('Median House Value')
plt.legend()
plt.show()

mv=float(input("enter median income="))
mv_array=np.array([[mv]])
predicted_price=model.predict(mv_array)
print("predicted price:",predicted_price)