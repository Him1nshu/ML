# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Create dataset (manually for now)
data = {
    "Hours": [1,2,3,4,5,6,7,8,9,10],
    "Marks": [35,40,50,55,60,65,70,80,85,95]
}

df = pd.DataFrame(data)

# Step 3: Separate input and output
X = df[["Hours"]]   # Independent variable
y = df["Marks"]     # Dependent variable

# Step 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create model
model = LinearRegression()

# Step 6: Train model
model.fit(X_train, y_train)

# Step 7: Predict
y_pred = model.predict(X_test)

# Step 8: Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Step 9: Plot
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression Model")
plt.show()
