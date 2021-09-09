import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random

#จำลองข้อมูล
x = rng.rand(50)*10
y = 2*x+rng.randn(50)

#Linear Regression Model
model = LinearRegression()

x2 = x.reshape(-1, 1)

#train Data
model.fit(x2, y)

#test set
xfit = np.linspace(-1,11)
xfitNew = xfit.reshape(-1,1)

# จับ xfit ไป test
yfit = model.predict(xfitNew)

#Analysis Model  & Result

plt.scatter(x,y)
plt.plot(xfit, yfit)
plt.show()