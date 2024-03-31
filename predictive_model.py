# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Function to get user inputs for house data
def get_house_data():
    bedrooms = int(input("Enter the number of bedrooms: "))
    sqft = int(input("Enter the square footage: "))
    return bedrooms, sqft

# Load dataset (you can replace this with a real dataset if available)
data = {
    'Bedrooms': [],
    'SqFt': [],
    'Price': []
}

num_samples = int(input("Enter the number of houses you want to use as data: "))

for i in range(num_samples):
    print(f"\nEnter details for house {i+1}:")
    bedrooms, sqft = get_house_data()
    price = int(input("Enter the price: "))
    data['Bedrooms'].append(bedrooms)
    data['SqFt'].append(sqft)
    data['Price'].append(price)

df = pd.DataFrame(data)

# Split data into features (X) and target variable (y)
X = df[['Bedrooms', 'SqFt']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'\nMean Squared Error: {mse}')

# Example prediction
print("\nEnter details for a new house:")
new_bedrooms, new_sqft = get_house_data()
new_data = {'Bedrooms': [new_bedrooms], 'SqFt': [new_sqft]}
new_df = pd.DataFrame(new_data)
predicted_price = model.predict(new_df)
print(f'\nPredicted Price for new house: ${predicted_price[0]}')