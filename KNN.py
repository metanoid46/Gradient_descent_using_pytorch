import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()

class KNN:
    def __init__(self, k=3):
        self.k = k
        
    def fit(self,x,y):
        self.x_train = x
        self.y_train = y
        
    def predict(self,X_test):
        predictions = []
        for i in range(len(X_test)):
            distance=X_test[i]-self.x_train
            distance=distance**2
            distance=distance.sum(axis=1)
            
            indexes=np.argsort(distance)
            indexes=indexes[:self.k]
            
            labels=self.y_train[indexes]
            counts=np.bincount(labels)
            prediction=np.argmax(counts)
            predictions.append(prediction)
          
            
        return predictions
    
    
iris = datasets.load_iris()
for i in range (1,10):
    
    k=5;
    X = iris.data
    y = iris.target

    x=np.array(X)
    y=np.array(y)

    indices = np.arange(len(X))  # Create index array
    np.random.shuffle(indices)  # Shuffle indices

    # Apply shuffled indices to both X and y
    X_shuffled, y_shuffled = X[indices], y[indices]

    # Split into training (80%) and testing (20%)
    X_train = X_shuffled[:int(len(X) * 0.8)]
    y_train = y_shuffled[:int(len(y) * 0.8)]
    X_test = X_shuffled[int(len(X) * 0.8):]
    y_test = y_shuffled[int(len(y) * 0.8):]

        
    print(np.isnan(X_train).sum()) 

    knn = KNN(k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)


    print("Predicted labels: ", y_pred, " \nTrue labels: ", y_test)
    # Compute Accuracy
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"KNN Model Accuracy: {accuracy:.2f}%")
