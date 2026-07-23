import pandas as pd
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("Train.csv")

print(df.head())
print(df.columns)
print(df["Item_MRP"].head())
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df[['Item_MRP']]
y = df['Item_Outlet_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained successfully")
predictions = model.predict(X_test)

print(predictions[:10])
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

plt.scatter(X_test, y_test, label="Actual")
plt.scatter(X_test, predictions, label="Predicted")

plt.xlabel("Item_MRP")
plt.ylabel("Item_Outlet_Sales")
plt.title("Sales Prediction")
plt.legend()
plt.show()
