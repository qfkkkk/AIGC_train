import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 混淆矩阵的数据
confusion_matrix = np.array([[21, 0, 0],
                             [1, 23, 1],
                             [0, 2, 6]])

# 创建混淆矩阵的图
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")

# 设置标题和坐标标签
plt.title("BiLSTM-Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.xticks(np.arange(3) + 0.5, ['0', '1', '2'])
plt.yticks(np.arange(3) + 0.5, ['0', '1', '2'], rotation=0)

# 显示图像
plt.show()
