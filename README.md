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

Mục tiêu của đồ án này này là thiết kế model dự đoán về khả năng vở nợ của khách hàng (default on loans) dựa vào dữ liệu nội bộ (của tổ chức) và bên ngoài của từng khách hàng
Metric của cuộc thi sử dụng gini stability metric:
<p style="text-align: center; font-size: small; font-style: italic;">gini = 2 × AUC − 1</p>
Và chuẩn đo cuối cùng là:
<p  style="text-align: center; font-size: small; font-style: italic">
stability metric=mean(gini) + 88.0 × min(0,a) − 0.5 × std(residuals)
</p>

Về dữ liệu, dữ liệu có một sự đa dạng ổn định, phù hợp cho mục tiêu của thiết kế model, trong đó có sự đặc biệt ở tập dữ liệu đó là được phân chia dựa trên giá trị `depth`: 
- depth = 0. Những đặc trưng tĩnh với từng `case_id` (gender, age ...)
- depth = 1. Những đặc trưng liên quan tới các bản ghi lịch sử của từng `case_id`  (previous applications, loans ...) được đánh số bởi `num_group1`
- depth = 2. Những đặc trưng cho biết thông tin chi tiết hơn đối với một vài trưng ở depth = 1 được đánh số bởi `num_group1` và `num_group2`
  
## Feature Engineering

Trước khi bắt đầu với việc gộp dữ liệu thì chúng tôi có biến đổi cột theo đúng định dạng được chỉ định ở từng đặc trưng ví dụ actualdpd_943P, amount_1115A ...
- `[P, A]` sang `Float64`
- `[M]` sang `String`
- `[D]` hoặc `date_decision` sang ` Date` 

Xử lý gộp data với việc sinh ra thêm 2 đặc trưng `month_decision` và `weekday_decision` từ `date_decision`
```python
df_base = (
        df_base
        .with_columns(
            month_decision = pl.col("date_decision").dt.month(),
            weekday_decision = pl.col("date_decision").dt.weekday(),
        )
        
    )
```
#### Xử lý các cột ngày tháng
- Các cột có hậu tố "D" được tính toán sự chênh lệch ngày so với cột date_decision và chuyển đổi thành số ngày (Float32).
- Các cột chứa năm (year) được tính toán sự chênh lệch năm so với năm của date_decision và chuyển đổi thành số nguyên (Int32).
- Cuối cùng, các cột không cần thiết như date_decision và MONTH được loại bỏ.
```python
def handle_dates(df):
        for col in df.columns:
            if col[-1]in ("D"):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
                df = df.with_columns(pl.col(col).cast(pl.Float32))
            elif "year" in col:
                df = df.with_columns(pl.col(col) - pl.col("date_decision").dt.year())
                df = df.with_columns(pl.col(col).cast(pl.Int32))
        df = df.drop("date_decision", "MONTH")
        return df
```
#### Lọc các cột không cần thiết
- Các cột có tỷ lệ giá trị null lớn hơn 85% được loại bỏ.
- Các cột chuỗi có tần suất giá trị duy nhất là 1 hoặc lớn hơn 200 cũng được loại bỏ.
```python
def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()
                if isnull > 0.85:
                    df = df.drop(col)
        
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        
        return df
```
- Xử lý các cột có độ tương quan trên 0.8

#### Tạo các biểu thức tổng hợp
Khi bắt đầu gộp dữ liệu, với bộ dữ liệu là depth 1 hoặc 2, quá trình sẽ thực hiện sinh các đặc trưng tổng hợp như sau:

- Đặc trưng có hậu tố `P` và `A` :  Sinh các cột số với các biểu thức tổng hợp như giá trị lớn nhất (max), giá trị đầu tiên (first), và giá trị trung bình (mean).
- Đặc trưng có hậu tố ` M` : Sinh các cột chuỗi với biểu thức mode (giá trị xuất hiện nhiều nhất) và giá trị lớn nhất (max).
- Đặc trưng có hậu tố `D` : Sinh các cột ngày tháng với các biểu thức như giá trị lớn nhất (max), giá trị đầu tiên (first), và giá trị trung bình (mean).
- Các cột có hậu tố `T, L` hoặc chứa `num_group` : Sinh các cột với biểu thức tổng hợp như giá trị lớn nhất (max) và giá trị đầu tiên (first).

#### Xử lý tạo mới đặc trưng
- Sinh thêm đặc trưng era từ first_birth_259D để xác định phạm vi giai đoạn:
```python 
df_base = df_base.with_columns(((pl.col("first_birth_259D") / 10).floor() * 10).alias("era").cast(pl.Int32))
```

### Những phát hiện
-  Bộ data bị lệch rất lớn 
-  `refreshdate_3813885D ` từ các tệp credit_bureau_a1*. Giá trị tối thiểu của nó cho mỗi ID trường hợp thường luôn bằng 3/1/2019 . Một lý do có thể là nó đã được sử dụng để điền vào các giá trị NaN. `refreshdate_3813885D` và `date_decision`  có một tương quan gần như hoàn hảo.





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
