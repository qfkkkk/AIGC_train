import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import make_scorer, f1_score
import spxy
plt.rcParams['font.family'] = ['SimHei'] # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus']=False
file_path = 'F:\光谱数据\模型训练数据\论文—苹果光谱数据//品种分类模型训练数据.csv'
data = pd.read_csv(file_path)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = spxy.random(X,y)
# 初始化XGBoost分类器
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 定义参数网格
param_grid = {
    'n_estimators': [100, 200, 300,400,500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2,0.5,0.8],
    'subsample': [0.5,0.6,0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9,1.0,1.2],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# 定义F1评分标准
f1_scorer = make_scorer(f1_score, average='macro')

# 使用网格搜索进行交叉验证
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring=f1_scorer, cv=5, verbose=1)
grid_search.fit(X_train, y_train)

# 输出最佳参数和最佳分数
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation F1 score: ", grid_search.best_score_)
