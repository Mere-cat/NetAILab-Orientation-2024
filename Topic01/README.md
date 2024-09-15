# Topic1 Assignment Report
Topic1: Machine Learning + sklearn

Assignment:
* hw_topic1.ipynb

## hw_topic1.ipynb
### 1. Missing Value Imputation
在第一個 TO-DO 中，我比較了三種補值方法（mean, knn, mice）的 MSE 以及執行時間，最終選擇 mice 作為本次作業的補值方法。

#### 1.1 Implementation

##### 儲存原始數據（未經缺值處理）
為計算三種補值方法的 MSE，需先儲存原始尚未丟失資料的數據。
```python
num_feat= df.drop("target",axis=1).select_dtypes("number").columns
original_values = df[num_feat] # original_values 為記錄的原始數據

# 隨機將200筆資料數值替換為缺失值
for col in num_feat:
    idx=np.random.choice(df.index,size=200,replace=False)
    df.loc[idx,col]=np.nan

missing_mask = df[num_feat].isna().to_numpy() # 紀錄失去的欄位

# id特徵不重要，刪除
df.drop("enrollee_id",axis=1,inplace=True)
original_values.drop("enrollee_id",axis=1,inplace=True) # 跟著 df 一起移除id

# performe frquent encoder (用該city出現的次數(數值型)來取代city(類別型))
city_freq=df["city"].value_counts(normalize=True)
df["city_freq"]=df["city"].map(city_freq)

original_values["city_freq"]=df["city"].map(city_freq)# 跟著 df 一起新增 city_freq
df.drop("city",axis=1,inplace=True)
```
##### 方法比較
```python
# 填補數值型特徵
########### TODO: 嘗試其他進階的填補方法(ex.MICE, KNN...) ##############
X = original_values.to_numpy() # original_values 從 dataframe 轉為 numpy array （與補值處理後的資料相同格式）

# Mean
start_time = time.time()
mean_imputer = SimpleFill(fill_method='mean')
X_filled_mean = mean_imputer.fit_transform(df[num_feat])
mean_mse = ((X_filled_mean[missing_mask] - X[missing_mask]) ** 2).mean()
mean_time = time.time() - start_time

# KNN
start_time = time.time()
knn_imputer = KNN(k=3)
X_filled_knn = knn_imputer.fit_transform(df[num_feat])
knn_mse = ((X_filled_knn[missing_mask] - X[missing_mask]) ** 2).mean()
knn_time = time.time() - start_time

# MICE
start_time = time.time()
mice_imputer = IterativeImputer()
X_filled_mice = mice_imputer.fit_transform(df[num_feat])
mice_mse = ((X_filled_mice[missing_mask] - X[missing_mask]) ** 2).mean()
mice_time = time.time() - start_time

# 結果
print(f"mean 的 MSE: {mean_mse}, 執行時間: {mean_time:.4f} 秒")
print(f"knn(k=3) 的 MSE: {knn_mse}, 執行時間: {knn_time:.4f} 秒")
print(f"mice 的 MSE: {mice_mse}, 執行時間: {mice_time:.4f} 秒")
#########################################################
```
#### 1.2 Result
![alt text](image/image1.png)

發現：
* KNN 的補值結果最接近原始資料，但也花費最多執行時間
* 簡單的平均值補值差原始資料最多，執行起來也最快

最終決定使用 mice，其表現不錯，也不會花多時間

### 2. Class Imbalance Problem
#### 2.1 Implementation
使用 ADASYN 製作更多少數類的資料點
```python
########### TODO: 嘗試其他resampling方法 #############
# smote = SMOTE()
# x_train, y_train = smote.fit_resample(x_train, y_train)
resampling = ADASYN()
x_train, y_train = resampling.fit_resample(x_train, y_train)
############################
```
#### 2.2 Result
![alt text](image/image2.png)

### 3. Classification Model
#### 3.1 Implementation
簡單比較三種模型（SVM, XGBoost, CatBoost）的準確率與 F1 Score
```python
########## TODO: 選擇你感興趣的model #############
# model = RandomForestClassifier()

# 定義一個函數來訓練和評估模型
def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, f1

# 要比較的模型
models = {
    "SVM": svm.SVC(probability=True),
    "XGBoost": XGBClassifier(),
    "CatBoost": CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiClass', verbose=0)
}

results = {}

# 訓練和評估
for model_name, model in models.items():
    accuracy, f1 = evaluate_model(model, x_train_array, y_train_array, x_test_array, y_test_array)
    results[model_name] = {"Accuracy": accuracy, "F1 Score": f1}

# 印出結果
for model_name, metrics in results.items():
    print(f"{model_name} - Accuracy: {metrics['Accuracy']}, F1 Score: {metrics['F1 Score']}")

model = svm.SVC(probability=True)
model.fit(x_train_array,y_train_array)
```

#### 3.2 Result
![alt text](image/image3.png)
這裡選擇準確率與 F1 Score 都略高的 SVM

模型的最終結果：
![alt text](image/image4.png)

### 4. Explainability
沒有 TO-DO，以當前模型去跑解釋器（shap.KernelExplainer 跑超級久，我也不知道到底跑了多久）

#### 4.1 Lime
![alt text](image/image5.png)

#### 4.2 SHAP
![alt text](image/SHAP1.png)

![alt text](image/SHAP2.png)

## 說明一下你的發現吧
1. 哪些特徵對預測結果影響大?
2. 那些特徵如何影響結果? (EX.某特徵越大，該培訓者越可能找其他工作)
3. 你如何理解這些發現?
