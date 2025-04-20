import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

# Data from https://www.un.org/development/desa/pd/sites/www.un.org.development.desa.pd/files/undesa_pd_2025_wfr-2024_advance-unedited.pdf
cols = ["x", "y"]

df = pd.read_csv("data.csv", names=cols)
df = df.rename(columns={"x": 0, "y": 1})   

# Correct: convert only columns, keep df as a DataFrame
df[0] = pd.to_numeric(df[0], errors='coerce')
df[1] = pd.to_numeric(df[1], errors='coerce')

# Now you can drop rows where either column has NaN
df = df.dropna(subset=[0, 1])

train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
x_train = train[[0]]
y_train = train[1]

lg_model = LinearRegression()
lg_model.fit(x_train,y_train)

y_pred = lg_model.predict(x_train)

plt.figure(figsize=(10,6))

plt.scatter(x_train, y_train,color='blue', label='Data points')

plt.plot(x_train, y_pred, color='red', label='Linear Regression')
plt.xlabel("x")
plt.ylabel("y")
plt.title('Linear Regression')
plt .legend()
plt.grid(True)
plt.show()



# To give users more options, we can use a barchart

x_valid = valid[[0]]
y_valid = valid[0]

knn = KNeighborsRegressor(n_neighbors=3)
knn_model = knn.fit(x_train, y_train)
y_pred = knn_model.predict(x_valid)

indices = np.arange(len(y_valid))

plt.figure(figsize=(10,6))

plt.bar(indices - 0.2, y_valid, width=0.4, label='Actual', color='blue')

plt.bar(indices + 0.2, y_pred, width=0.4, label='Predicted', color='Red')

# Add labels and legend
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.show()