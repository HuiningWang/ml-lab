import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.25, n_iterations=100):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 初始化权重和偏置
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                linear_output = sum(self.weights[j] * X[i][j] for j in range(n_features)) + self.bias
                prediction = 1 if linear_output >= 0 else 0
                # 更新权重和偏置
                update = self.learning_rate * (y[i] - prediction)
                for j in range(n_features):
                    self.weights[j] += update * X[i][j]
                self.bias += update

    def predict(self, X):
        linear_output = sum(self.weights[j] * X[j] for j in range(len(X))) + self.bias
        return 1 if linear_output >= 0 else 0


# 定义 AND 问题的数据集
X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]

# 定义 OR 问题的数据集
X_or = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_or = [0, 1, 1, 1]

# 训练感知机解决 AND 问题
perceptron_and = Perceptron()
perceptron_and.fit(np.array(X_and), np.array(y_and))
print("AND 问题的权重：", perceptron_and.weights, "偏置：", perceptron_and.bias)
print("AND 问题预测结果：")
for x in X_and:
    print(f"输入 {x}，预测结果 {perceptron_and.predict(x)}")

# 训练感知机解决 OR 问题
perceptron_or = Perceptron()
perceptron_or.fit(np.array(X_or), np.array(y_or))
print("\nOR 问题的权重：", perceptron_or.weights, "偏置：", perceptron_or.bias)
print("OR 问题预测结果：")
for x in X_or:
    print(f"输入 {x}，预测结果 {perceptron_or.predict(x)}")