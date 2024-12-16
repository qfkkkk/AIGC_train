import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import pro
import spxy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras_self_attention import SeqSelfAttention
from keras.callbacks import ModelCheckpoint
from keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Flatten,Attention
from keras.models import Model
from keras.regularizers import l1
from keras.layers import MultiHeadAttention
import warnings
from keras.optimizers import Adam
from imblearn.over_sampling import SMOTE
# 忽略特定类型的TensorFlow警告
warnings.filterwarnings('ignore', category=DeprecationWarning)
# 使用pandas加载CSV文件
#file_path = 'F:\光谱数据\模型训练数据\H100 苹果数据//苹果H100 光谱数据_分类_200 .csv'
# 加载数据
file_path = 'F:\光谱数据\模型训练数据\论文—苹果光谱数据//洛川苹果红富士分类.csv'
data = pd.read_csv(file_path)

# 分离特征和标签
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 对标签进行独热编码
num_classes = len(np.unique(y))
y_one_hot = to_categorical(y, num_classes=num_classes)

# 数据预处理：标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 应用你的特殊处理，假设是一种光谱处理
X_SG = pro.SNV(X_scaled)

# 划分数据集（在应用SMOTE之前进行）
X_train, X_test, y_train_one_hot, y_test_one_hot = spxy.spxy(X_SG, y_one_hot, test_size=0.2)

# 将one-hot编码的标签转换回原始的类别编码以适应SMOTE
y_train = np.argmax(y_train_one_hot, axis=1)

# 创建SMOTE对象并对训练集进行过采样
sm = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = sm.fit_resample(X_train, y_train)

# 由于SMOTE改变了样本数量，需要将标签重新转换为one-hot编码
y_train_resampled_one_hot = to_categorical(y_train_resampled, num_classes=num_classes)

# 调整输入数据的形状以适应LSTM模型，对过采样后的训练数据进行操作
X_train_resampled = np.reshape(X_train_resampled, (X_train_resampled.shape[0], 1, X_train_resampled.shape[1]))
# 对测试数据也执行相同的重塑操作
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# 现在，X_train_resampled和X_test已经准备好用于模型训练和评估
#############  双向LSTM层 #######
def create_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    # 双向LSTM层
    x = Bidirectional(LSTM(30, return_sequences=True, activation='tanh'))(inputs)
    x = Bidirectional(LSTM(30, return_sequences=True, activation='tanh'))(x)
    # 多头注意力机制
    # 注意：为了使用MultiHeadAttention，我们需要确保特征维度能够被头数整除。
    query_encoding = Dense(30)(x)
    value_encoding = Dense(30)(x)
    attn_output = MultiHeadAttention(num_heads=2, key_dim=8)(query_encoding, value_encoding)
    # 可能需要根据模型的具体要求调整输出层前的层

    x = Flatten()(attn_output)  # 根据需要调整，例如使用Glo balAveragePooling1D等
    x = Dropout(0.25)(x)
    outputs = Dense(num_classes, activation='softmax', activity_regularizer=l1(0.0001))(x)  #L1正则化技术
    model = Model(inputs=inputs, outputs=outputs)
    return model
# 自定义学习率的修改
custom_lr = 0.0001  # 你可以根据需要修改这个学习率
adam_optimizer = Adam(learning_rate=custom_lr)

model = create_model((1, X_train_resampled.shape[2]), num_classes)

model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
# 设置ModelCheckpoint
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy', mode='max')
#使用早停法
#early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=50, verbose=1, mode='min')
# 在fit函数中加入callbacks参数
history = model.fit(X_train_resampled, y_train_resampled_one_hot, epochs=1000, batch_size=128, validation_data=(X_test, y_test_one_hot), callbacks=[checkpoint])
# 加载最佳模型
model = load_model('best_model.h5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
# 评估模型
loss, accuracy = model.evaluate(X_test, y_test_one_hot)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
# 假设你的模型输出是概率，首先将它们转换为类别标签
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test_one_hot, axis=1)
# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred_classes)
# 可视化混淆矩阵
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
# 绘制准确率曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# 绘制损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()