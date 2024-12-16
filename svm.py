import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import spxy
import pro
# 读取数据
file_path = 'F:\光谱数据\模型训练数据\论文—苹果光谱数据//洛川苹果红富士分类2.csv'  # 替换为实际文件路径
data = pd.read_csv(file_path)

# 数据预处理
X = data.iloc[:, :-1].values  # 提取特征列
y = data.iloc[:, -1].values  # 提取标签列

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_SG=pro.SNV(X_scaled)
# 划分训练集和测试集
X_train, X_test, y_train, y_test = spxy.spxy(X_SG, y, test_size=0.2)

# 使用SVM分类器进行训练
svm_classifier = SVC(kernel='rbf')  # 使用线性核
svm_classifier.fit(X_train, y_train)

# 预测
y_pred = svm_classifier.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
