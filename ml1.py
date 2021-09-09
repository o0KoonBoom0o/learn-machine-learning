from sklearn import datasets

iris = datasets.load_iris()

print(iris['data'][0:10])