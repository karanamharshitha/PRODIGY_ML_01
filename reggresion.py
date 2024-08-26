import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data = pd.read_csv('housing_data.csv')

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())
# Select features (independent variables)
X = data[['square_footage', 'bedrooms', 'bathrooms']]

# Select the target (dependent variable)
y = data['price']
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create an instance of the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)
# Predict the house prices for the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Plot predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.show()
# Example: Predict the price of a house with 2000 sq ft, 3 bedrooms, and 2 bathrooms
new_data = np.array([[2000, 3, 2]])
predicted_price = model.predict(new_data)

print(f'Predicted Price: {predicted_price[0]}')