from padelpy import from_smiles
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer

os.environ["JAVA_TOOL_OPTIONS"] = "-Dfile.encoding=UTF-8"

file_path0 = 'basic_smiles.xlsx'
df = pd.read_excel(file_path0)
smiles_list = (df['SMILES'].dropna().astype(str).tolist())

# save descriptors to a CSV file
_ = from_smiles(smiles_list, output_csv='basic_descriptors.csv')

file_path = 'autodevice-results1215.xlsx'
df = pd.read_excel(file_path)
num_list = df['num'].tolist()
com1 = df['coma-l'].tolist()
com2 = df['comP1-P8'].tolist()
com3 = df['com1-20'].tolist()

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

# --------------------------
# 配置
# --------------------------
RANDOM_STATE = 42
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3
N_ITER_BAYES = 20  # 贝叶斯搜索迭代次数

# --------------------------
# 读取数据
# --------------------------
X = pd.read_csv("all_des.csv", header=None).values
y = pd.read_excel("autodevice-results1215.xlsx")['average'].apply(lambda x: 1 if x >= 50000 else 0).values

# --------------------------
# 自定义 PR AUC 评分函数
# --------------------------
def pr_auc_score(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    return auc(recall, precision)

pr_scorer = make_scorer(pr_auc_score, needs_proba=True)

# --------------------------
# 外层交叉验证
# --------------------------
outer_cv = StratifiedKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_STATE)
fold_idx = 0

# 保存各模型最佳信息
best_models = {
    "xgb": {"model": None, "roc_auc": -1, "pr_auc": -1},
    "rf":  {"model": None, "roc_auc": -1, "pr_auc": -1},
    "lr":  {"model": None, "roc_auc": -1, "pr_auc": -1},
}

# 保存每折曲线
all_fold_curves = []

for train_idx, test_idx in outer_cv.split(X, y):
    fold_idx += 1
    print(f"\n--- Outer fold {fold_idx}/{N_SPLITS_OUTER} ---")

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_pos_train = int(np.sum(y_train == 1))
    n_neg_train = len(y_train) - n_pos_train
    n_pos_test = int(np.sum(y_test == 1))
    n_neg_test = len(y_test) - n_pos_test

    print(f"Train pos/neg: {n_pos_train} {n_neg_train} | Test pos/neg: {n_pos_test} {n_neg_test}")

    if n_pos_train == 0:
        print("WARNING: no positive samples in training fold. Skipping fold.")
        continue

    # --------------------------
    # 过采样
    # --------------------------
    if n_pos_train >= 2:
        k_neighbors = min(5, n_pos_train - 1)
        sampler = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    else:
        sampler = RandomOverSampler(random_state=RANDOM_STATE)

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    print(f"Resampled train pos/neg: {int(np.sum(y_res==1))} {len(y_res)-int(np.sum(y_res==1))}")

    # --------------------------
    # 三模型贝叶斯调参
    # --------------------------

    # XGBoost
    xgb_space = {
        'max_depth': Integer(3, 10),
        'learning_rate': Real(0.01, 0.3, 'log-uniform'),
        'n_estimators': Integer(50, 500),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'gamma': Real(0, 5),
        'min_child_weight': Integer(1, 10)
    }
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE)
    search_xgb = BayesSearchCV(
        estimator=xgb_model,
        search_spaces=xgb_space,
        n_iter=N_ITER_BAYES,
        scoring=pr_scorer,
        cv=N_SPLITS_INNER,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE
    )
    search_xgb.fit(X_res, y_res)
    xgb_best = search_xgb.best_estimator_

    # RandomForest
    rf_space = {
        'n_estimators': Integer(50, 500),
        'max_depth': Integer(3, 20),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 5)
    }
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE)
    search_rf = BayesSearchCV(
        estimator=rf_model,
        search_spaces=rf_space,
        n_iter=N_ITER_BAYES,
        scoring=pr_scorer,
        cv=N_SPLITS_INNER,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE
    )
    search_rf.fit(X_res, y_res)
    rf_best = search_rf.best_estimator_

    # LogisticRegression
    lr_space = {
        'C': Real(0.01, 100, 'log-uniform'),
        'max_iter': Integer(500, 2000)
    }
    lr_model = LogisticRegression(random_state=RANDOM_STATE, solver='liblinear')
    search_lr = BayesSearchCV(
        estimator=lr_model,
        search_spaces=lr_space,
        n_iter=N_ITER_BAYES,
        scoring=pr_scorer,
        cv=N_SPLITS_INNER,
        n_jobs=-1,
        verbose=0,
        random_state=RANDOM_STATE
    )
    search_lr.fit(X_res, y_res)
    lr_best = search_lr.best_estimator_

    # --------------------------
    # 预测 & 计算 AUC
    # --------------------------
    models = {'xgb': xgb_best, 'rf': rf_best, 'lr': lr_best}
    fold_curves = {}

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        pr_auc = auc(recall, precision)

        print(f"{name.upper()}  ROC AUC = {roc_auc:.4f}, PR AUC = {pr_auc:.4f}")

        # 更新全局最佳
        if pr_auc > best_models[name]['pr_auc']:
            best_models[name]['model'] = model
            best_models[name]['roc_auc'] = roc_auc
            best_models[name]['pr_auc'] = pr_auc

        # 保存 fold 曲线
        fold_curves[name] = {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall}

    all_fold_curves.append(fold_curves)

# --------------------------
# 保存最佳模型
# --------------------------
for name, info in best_models.items():
    save_path = f"best_{name}_model.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(info['model'], f)
    print(f"已保存 {save_path} | PR AUC={info['pr_auc']:.4f}")

