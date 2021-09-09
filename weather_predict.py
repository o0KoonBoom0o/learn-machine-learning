import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Weather.csv")

# train & test set
x = dataset["MinTemp"].values.reshape(-1,1)
y = dataset["MaxTemp"].values.reshape(-1,1)

xTrain, xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2,random_state=0)

# Training
model = LinearRegression()
model.fit(xTrain,yTrain)

# Test พยากรอุณหูมิสูงสุด จากอุณหภูมิิต่ำสุด
yPredict = model.predict(xTest)

# compare get in data form
df = pd.DataFrame({'Actually' : yTest.flatten(), 'Predict': yPredict.flatten()})

barChart = df.head(20)
barChart.plot(kind='bar',figsize=(16,10))
plt.show()