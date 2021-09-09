from sklearn import tree

model = tree.DecisionTreeClassifier()

feature = [ [35000, 80000, 0, 0],
            [25000, 10000, 1, 0],
            [63000, 20000, 2, 2],
            [96000, 50000, 5, 1],
            [44000, 60000, 1, 0]
          ]

label = [10, 1, 1, 10, 10]

model.fit(feature,label)

print(model.predict([[96000, 20000, 5, 2]]))