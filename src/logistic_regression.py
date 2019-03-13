import numpy as np
from scipy import linalg

class LogisticRegression:
  def __init__(self, abs=0.1, seed=0, max_iteration=5):
    self.weight = None
    self.abs = abs
    self.random_state_generator = np.random.RandomState(seed)
    self.max_iteration = max_iteration

  def sigmoid(self, x):
    return 1 / (1 + np.exp(-x))

  def train(self, feature, target):
    X = np.c_[np.ones(feature.shape[0]), feature]
    # weightをrandomで生成
    self.weight = self.random_state_generator.randn(X.shape[1])
    diff_value = np.inf
    iteration_count = 0
    # 逐次的にw(k)を求める
    while diff_value > self.abs and iteration_count < self.max_iteration:
      # 式中で複数回現れる項を計算
      Xw = np.dot(X, self.weight)
      sigmoid_Xw = self.sigmoid(Xw)
      # A(w(k+1) - w(k)) = Bの形を作成して解く
      sigmoid_mul = np.dot(sigmoid_Xw.T, (1 - sigmoid_Xw))
      A = sigmoid_mul * np.dot(X.t, X)
      
      B = np.dot(X.T, (np.array(target) - sigmoid_Xw))
      # w(k+1) - w(k)
      diff = linalg.solve(A, B)
      self.weight += diff
      # 条件判定のためのカウント
      iteration_count += 1
      diff_value = diff.mean()

  def predict(self, input_data):
    if self.weight is None:
      raise ValueError('model need to be trained')

    X = np.c_[np.ones(input_data.shape[0]), input_data]
    result = self.sigmoid(np.dot(X, self.weight))
    return np.where(result >= 0.5, 1, 0)
