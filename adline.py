import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class ADLINE:
    def __init__(self, learning_rate=0.01, n_iterations=100, mode='online'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.mode = mode  # 'online' 或 'batch'
        self.weights = None
        self.bias = None
        self.costs = []  # 记录损失变化
        self.weight_history = []  # 记录权重变化，用于绘制优化轨迹

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            if self.mode == 'online':
                cost = 0
                for i in range(n_samples):
                    linear_output = np.dot(self.weights, X[i]) + self.bias
                    error = y[i] - linear_output
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    self.weight_history.append(self.weights.copy())  # 记录在线模式下的权重轨迹
                    cost += 0.5 * error ** 2
                self.costs.append(cost / n_samples)
            elif self.mode == 'batch':
                linear_outputs = np.dot(X, self.weights) + self.bias
                errors = y - linear_outputs
                self.weights += self.learning_rate * np.dot(X.T, errors)
                self.bias += self.learning_rate * np.sum(errors)
                self.weight_history.append(self.weights.copy())  # 记录批量模式下的权重轨迹
                cost = 0.5 * np.sum(errors ** 2) / n_samples
                self.costs.append(cost)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict_binary(self, X, threshold=0.5):
        return (self.predict(X) >= threshold).astype(int)

# 定义AND问题的数据集
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

# 定义OR问题的数据集
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

# 训练ADLINE解决AND问题（在线模式）
adline_and_online = ADLINE(mode='online', learning_rate=0.05, n_iterations=50)
adline_and_online.fit(X_and, y_and)

# 训练ADLINE解决AND问题（批量模式）
adline_and_batch = ADLINE(mode='batch', learning_rate=0.05, n_iterations=50)
adline_and_batch.fit(X_and, y_and)

# 训练ADLINE解决OR问题（在线模式）
adline_or_online = ADLINE(mode='online', learning_rate=0.05, n_iterations=50)
adline_or_online.fit(X_or, y_or)

# 训练ADLINE解决OR问题（批量模式）
adline_or_batch = ADLINE(mode='batch', learning_rate=0.05, n_iterations=50)
adline_or_batch.fit(X_or, y_or)

# ========== 可视化损失曲线 ==========
plt.figure(figsize=(12, 4))

# AND问题损失曲线
plt.subplot(1, 2, 1)
plt.plot(adline_and_online.costs, label='Online (AND)')
plt.plot(adline_and_batch.costs, label='Batch (AND)')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('ADLINE Loss - AND Problem')
plt.legend()

# OR问题损失曲线
plt.subplot(1, 2, 2)
plt.plot(adline_or_online.costs, label='Online (OR)')
plt.plot(adline_or_batch.costs, label='Batch (OR)')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('ADLINE Loss - OR Problem')
plt.legend()

plt.tight_layout()
plt.show()

# ========== 可视化决策边界 ==========
plt.figure(figsize=(12, 4))

# AND问题决策边界
plt.subplot(1, 2, 1)
plt.scatter(X_and[:, 0], X_and[:, 1], c=y_and, cmap='viridis')
x = np.linspace(-0.5, 1.5, 100)
y_online = (0.5 - adline_and_online.weights[0] * x - adline_and_online.bias) / adline_and_online.weights[1]
y_batch = (0.5 - adline_and_batch.weights[0] * x - adline_and_batch.bias) / adline_and_batch.weights[1]
plt.plot(x, y_online, label='Online Decision Boundary (AND)')
plt.plot(x, y_batch, label='Batch Decision Boundary (AND)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('ADLINE Decision Boundary - AND Problem')
plt.legend()
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

# OR问题决策边界
plt.subplot(1, 2, 2)
plt.scatter(X_or[:, 0], X_or[:, 1], c=y_or, cmap='viridis')
x = np.linspace(-0.5, 1.5, 100)
y_online = (0.5 - adline_or_online.weights[0] * x - adline_or_online.bias) / adline_or_online.weights[1]
y_batch = (0.5 - adline_or_batch.weights[0] * x - adline_or_batch.bias) / adline_or_batch.weights[1]
plt.plot(x, y_online, label='Online Decision Boundary (OR)')
plt.plot(x, y_batch, label='Batch Decision Boundary (OR)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('ADLINE Decision Boundary - OR Problem')
plt.legend()
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)

plt.tight_layout()
plt.show()

# ========== 可视化ADLINE误差曲面（以AND问题为例） ==========
def adline_error(w0, w1, X, y, bias=0):
    n_samples = X.shape[0]
    error = 0
    for i in range(n_samples):
        linear_output = w0 * X[i, 0] + w1 * X[i, 1] + bias
        error += 0.5 * (y[i] - linear_output) ** 2
    return error / n_samples

# 生成权重网格
w0_range = np.linspace(-1, 3, 50)
w1_range = np.linspace(-1, 2, 50)
w0_mesh, w1_mesh = np.meshgrid(w0_range, w1_range)

# 计算每个网格点的误差
error_mesh = np.zeros_like(w0_mesh)
for i in range(w0_mesh.shape[0]):
    for j in range(w0_mesh.shape[1]):
        error_mesh[i, j] = adline_error(w0_mesh[i, j], w1_mesh[i, j], X_and, y_and, adline_and_batch.bias)

# 绘制误差曲面和优化轨迹
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制误差曲面
surf = ax.plot_surface(w0_mesh, w1_mesh, error_mesh, cmap='viridis', alpha=0.8)
fig.colorbar(surf, shrink=0.5, aspect=5)

# 提取批量模式下的权重轨迹（在线模式可同理添加）
weights_history_batch = np.array(adline_and_batch.weight_history)
if len(weights_history_batch) > 0:
    w0_hist = weights_history_batch[:, 0]
    w1_hist = weights_history_batch[:, 1]
    error_hist = [adline_error(w0, w1, X_and, y_and, adline_and_batch.bias) for w0, w1 in zip(w0_hist, w1_hist)]
    ax.plot(w0_hist, w1_hist, error_hist, 'r-', marker='o', markersize=5, label='Batch Optimization Trajectory')

ax.set_xlabel('w0')
ax.set_ylabel('w1')
ax.set_zlabel('E[w]')
ax.set_title('ADLINE Error Surface and Optimization Trajectory (AND Problem)')
ax.legend()

plt.show()

# 打印权重、偏置和预测结果
print("=== AND 问题 ===")
print("在线模式权重：", adline_and_online.weights, "偏置：", adline_and_online.bias)
print("批量模式权重：", adline_and_batch.weights, "偏置：", adline_and_batch.bias)
print("\n=== OR 问题 ===")
print("在线模式权重：", adline_or_online.weights, "偏置：", adline_or_online.bias)
print("批量模式权重：", adline_or_batch.weights, "偏置：", adline_or_batch.bias)

print("\n=== AND 问题预测（在线模式）===")
for x in X_and:
    print(f"输入 {x}，预测概率 {adline_and_online.predict(x):.2f}，二分类预测 {adline_and_online.predict_binary(x)}")

print("\n=== AND 问题预测（批量模式）===")
for x in X_and:
    print(f"输入 {x}，预测概率 {adline_and_batch.predict(x):.2f}，二分类预测 {adline_and_batch.predict_binary(x)}")

print("\n=== OR 问题预测（在线模式）===")
for x in X_or:
    print(f"输入 {x}，预测概率 {adline_or_online.predict(x):.2f}，二分类预测 {adline_or_online.predict_binary(x)}")

print("\n=== OR 问题预测（批量模式）===")
for x in X_or:
    print(f"输入 {x}，预测概率 {adline_or_batch.predict(x):.2f}，二分类预测 {adline_or_batch.predict_binary(x)}")