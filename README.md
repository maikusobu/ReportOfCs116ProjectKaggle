<p align="center">
  <h3 align="center">Báo cáo đồ án cho môn lập trình máy học cho python - CS112</h3>
  <p align="center">
    Tác giả
  </p>
</p>

## Mục lục

- [Introduction](#introduction)
- [Feature engineer](#feature-engineer)
- [Model](#model)

## Giới thiệu

Từ kinh nghiệm của tôi trong việc áp dụng Machine Learning, tôi nhận thấy rằng cuộc thi Home Credit - Credit Risk Model Stability là một trong những thử thách phức tạp nhất cho tôi cho đến nay. Dữ liệu được cung cấp là một tập hợp lớn, chứa nhiều thông tin từ nhiều nguồn và phản ánh thực tế về tài chính của các cá nhân. Để giải quyết bài toán này, cần có kiến thức về tài chính, khả năng phân tích, và khả năng hiểu các dữ liệu trong lĩnh vực ngân hàng.

Dựa trên những hiểu biết của tôi về lĩnh vực tài chính cũng như những bài toán machine learning tương tự. Tôi cho rằng có hai điều rất quan trọng để xây dựng tốt mô hình cho cuộc thi này : 

1. Thực hiện tốt Feature Engineering
2. Sử dụng kết hợp đa các mô hình Machine learning.


## Data and Feature Engineering

## Base Model

Chúng tôi dựa trên mô hình được công bố trên notebook : Credit Risk Prediction with LightGBM and C.

Mô hình gốc sử dụng hai mô hình chính là **lgboost** và **catboost**. Sau đó sử dụng **VotingModel** để kết hợp các kết quả dự đoán. Các chi tiết được mô tả cụ thể bên dưới đây : 

### Về cách khởi tạo các mô hình

Đầu tiên, lược bỏ cột 'target', 'case_id' và 'week_num' :

```python 
X = df_train.drop(columns=["target", "case_id", "week_num"])
y = df_train["target"]
```

Như đã trình bày, tác giả sử dụng hai loại mô hình chính là lgboost và catboost. Tuy nhiên, một điểm cần lưu ý là với mỗi loại mô hình, có nhiều mô hình được tạo ra (instance của class lagboost và catboost) để predict cho từng week_num một, ta có hai danh sách lưu các mô hình cho hai loại mô hình:

```python
fitted_models_cat = []
fitted_models_lgb = []
```


Chia dữ liệu train theo các group, với mỗi group được xác định bởi giá trị của cột 'week_num', tức là hàng nào có cùng 'week_num' thì đưa vào chung một group : 

```python 
weeks = df_train["week_num"]
for idx_train, idx_valid in cv.split(X, y, groups=weeks):
```


Sau đó, đối với mỗi dữ liệu phân theo week_num, ta tạo và huấn luyện một mô hình lgboost và catboost:

Đối với castboost, các hyper param được giữ cố định:

```python 

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_valid, y_valid, cat_features=cat_cols)

    clf = CatBoostClassifier(
        best_model_min_trees = 1000,
        boosting_type = "Plain",
        eval_metric = "AUC",
        iterations = est_cnt,
        learning_rate = 0.05,
        l2_leaf_reg = 10,
        max_leaves = 64,
        random_seed = 42,
        task_type = "GPU",
        use_best_model = True
    )
    clf.fit(train_pool, eval_set=val_pool, verbose=False)
    fitted_models_cat.append(clf)
```

Đối với lgboost, hyper param được dùng khác nhau cho mỗi lần lặp chẵn hoặc lẻ:

```python
    if iter_cnt % 2 == 0:
        model = lgb.LGBMClassifier(**params1)
    else:
        model = lgb.LGBMClassifier(**params2)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[lgb.log_evaluation(100), lgb.early_stopping(100)],
    )
    fitted_models_lgb.append(model)
```

Hyper param cho catboost là : 

```python 
params1 = {
    "boosting_type": "gbdt",
    "colsample_bynode": 0.8,
    "colsample_bytree": 0.8,
    "device": device,
    "extra_trees": True,
    "learning_rate": 0.05,
    "l1_regularization": 0.1,
    "l2_regularization": 10,
    "max_depth": 20,
    "metric": "auc",
    "n_estimators": 2000,
    "num_leaves": 64,
    "objective": "binary",
    "random_state": 42,
    "verbose": -1,
}

params2 = {
    "boosting_type": "gbdt",
    "colsample_bynode": 0.8,
    "colsample_bytree": 0.8,
    "device": device,
    "extra_trees": True,
    "learning_rate": 0.03,
    "l1_regularization": 0.1,
    "l2_regularization": 10,
    "max_depth": 16,
    "metric": "auc",
    "n_estimators": 2000,
    "num_leaves": 54,
    "objective": "binary",
    "random_state": 42,
    "verbose": -1,
}
```

Sau khi huấn luyện mô hình, sử dụng một VotingModel để đưa ra dự đoán, VotingModel chỉ đơn giản là lấy mean của các dự đoán : 

```python 
class VotingModel(BaseEstimator, ClassifierMixin):
    ...
    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
```

## Các lần thử



| What     | Result   | 
| -------- | -------- |
| random_seed = 3107     | 0.57     |
| best_model_min_trees = 1200 | MLE |
| auto_weight and class_weight| 0.544|
| add xgboost| MLE |
