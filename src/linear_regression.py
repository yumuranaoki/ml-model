import numpy as np
from scipy import linalg

class LinearRegression:
  def __init__(self):
    self.weight = None

  def train(self, feature, target):
    if feature.shape[0] != target.shape[0]:
      raise ValueError('data need to be reshaped')

    # featureからXを用意する
    X = np.c_[np.ones(feature.shape[0]), feature]
    
    # A*w = B からwを解く
    A = np.dot(X.T, X)
    B = np.dot(X.T, target)
    weight = linalg.solve(A, B)

    # self.weightにwを突っ込む
    self.weight = weight


  def predict(self, input_data):
    if self.weight is None:
      raise ValueError('model need to be trained')
    
    X = np.c_[np.ones(input_data.shape[0]), input_data]

    prediction = np.dot(X, self.weight)
    return prediction
    

