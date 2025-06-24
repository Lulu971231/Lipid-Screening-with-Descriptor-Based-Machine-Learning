import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import xgboost as xgb
from padelpy import from_smiles

############################1、Data preparation and descriptor generation############################
file_path = 'basic_smiles.xlsx'
df = pd.read_excel(file_path)
smiles_list = df['smiles'].tolist()

# save descriptors to a CSV file
_ = from_smiles(smiles_list, output_csv='basic_descriptors.csv')

############################2、Component combination descriptor splicing############################
file_path = '结构的基元组分.xlsx'
df = pd.read_excel(file_path)
num_list = df['产物编号'].tolist()
com1 = df['基元1'].tolist()
com2 = df['基元2'].tolist()
com3 = df['基元3'].tolist()

file_path2 = 'basic_descriptors.csv'
data = pd.read_csv(file_path2)
num2 = data['Name'].tolist()
single_des = data.iloc[0:,1:]
single_des = np.array(single_des)

descriptor = []
for i in range(0,len(num_list)):
    everydes = []
    zu1 = com1[i]
    zu2 = com2[i]
    zu3 = str(com3[i])

    for j in range(0,len(num2)):
        if zu1 == num2[j]:
            
            for k in single_des[j]:
                everydes.append(k)
            break
    for j in range(0,len(num2)):
        if zu2 == num2[j]:
            
            for k in single_des[j]:
                everydes.append(k)
                
            break
    for j in range(0,len(num2)):
        if zu3 == num2[j]:
            
            for k in single_des[j]:
                everydes.append(k)
            break
    descriptor.append(everydes)
   

descriptor = np.array(descriptor)

np.savetxt('all_des.csv',descriptor,delimiter=',',fmt='%.8f')

############################3、Model training############################
# 读取实验数据
input_data=pd.read_csv("all_des.csv",header=None)
file_path = '结构的基元组分.xlsx'
data = pd.read_excel(file_path)
smiles_list = data['average'].values

# 分离特征和标签
X = input_data.iloc[:,0:]
X = np.array(X)
y = data['average'].apply(lambda x: 1 if x >= 150000 else 0)

# 使用SMOTE平衡数据
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 训练模型
seed = 42
xgb_model = XGBClassifier(random_state=seed, tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

xgb_model.fit(X_train, y_train)

# 预测概率
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

xgb_model.save_model('xgboost_classification_150000_model_0328.json')
'''
test_data=pd.read_csv("test_des.csv",header=None)
test_data = test_data.iloc[:,0:]
test_result = xgb_model.predict(test_data)
print(test_result)
'''
# 计算ROC和PR曲线
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs)

xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_probs)

xgb_auc = auc(xgb_fpr, xgb_tpr)

# 评估模型
#xgb_pred = xgb_model.predict(X_test)

#xgb_accuracy = accuracy_score(y_test, xgb_pred)

#print(f"XGBoost Accuracy: {xgb_accuracy}")

############################4、Virtual library combination and screening############################
from collections import Counter

file_path = 'basic_descriptors0702.csv'
data = pd.read_csv(file_path)

# 生成虚拟数据示例
component_name1 = data['Name'][0:36]
component_1 = data.iloc[0:36,1:].values
print(component_name1)

component_name2 = data['Name'][68:81]
component_2 = data.iloc[68:81,1:].values
print(component_name2)

component_name3 = data['Name'][36:68]
component_3 = data.iloc[36:68,1:].values
print(component_name3)
# 生成所有可能的三类组分组合
virtual_library_combinations = list(product(component_name1, component_name2, component_name3))

# 准备虚拟库的特征矩阵
num_virtual_lipids = len(virtual_library_combinations)
virtual_library_features = np.zeros((num_virtual_lipids, component_1.shape[1] + component_2.shape[1] + component_3.shape[1]))

for i, (hg, lk, tl) in enumerate(virtual_library_combinations):
    hg_idx = np.where(component_name1 == hg)[0][0]
    lk_idx = np.where(component_name2 == lk)[0][0]
    tl_idx = np.where(component_name3 == tl)[0][0]
    virtual_library_features[i] = np.hstack((component_1[hg_idx], component_2[lk_idx], component_3[tl_idx]))

# 初始化存储顶级脂质候选物的列表
top_lipids_indices = []
seeds = np.random.randint(0, 10000, size=1000)
# 多次运行模型筛选顶级脂质
for seed in seeds:
    xgb_model = XGBClassifier(random_state=seed, tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)
    xgb_model.fit(X_train, y_train)
    predictions = xgb_model.predict_proba(virtual_library_features)[:, 1]
    #top_50_indices = np.argsort(predictions)[-50:]
    top_50_indices = np.argsort(predictions)
    top_50_indices = [idx for idx in top_50_indices if predictions[idx] > 0.6]
    #top_50_indices = np.argsort(predictions)[-50:]
    top_lipids_indices.extend(top_50_indices)

top_lipids_counter = Counter(top_lipids_indices)

# 提取顶级脂质的组合
top_lipids_combinations = np.array(virtual_library_combinations)[top_lipids_indices]

# 统计每个组件的出现频率
head_group_counter = Counter(top_lipids_combinations[:, 0])
linker_counter = Counter(top_lipids_combinations[:, 1])
tail_counter = Counter(top_lipids_combinations[:, 2])

# 获取每类组分中出现频率最高的前几名
top_n = 10
top_head_groups = head_group_counter.most_common(top_n)
top_linkers = linker_counter.most_common(top_n)
top_tails = tail_counter.most_common(top_n)

print("Top component1:")
for component, freq in top_head_groups:
    print(f"Component: {component}, Frequency: {freq}")

print("\nTop component2:")
for component, freq in top_linkers:
    print(f"Component: {component}, Frequency: {freq}")

print("\nTop component3:")
for component, freq in top_tails:
    print(f"Component: {component}, Frequency: {freq}")

print("\nTop 50 Lipids and Their Frequencies:")
for lipid, freq in top_lipids_counter.most_common(50):
    combination = virtual_library_combinations[lipid]
    print(f"Lipid: {combination}, Frequency: {freq}")
