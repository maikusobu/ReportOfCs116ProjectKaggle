# Báo cáo đồ án Lập trình Python cho Máy Học (Python Programming for Machine LEarning) - CS116
## Tác giả

| STT | MSSV     | Họ và Tên                    | Github                                            | Email                    |
| --- | -------- | ---------------------------- | ------------------------------------------------- | ------------------------ |
| 1   | 22520801 | Nguyễn Tấn Lợi               | [maikusobu](https://github.com/maikusobu)         | <22520801@gm.uit.edu.vn> |
| 2   | 22520847 | Nguyễn Đặng Đức Mạnh         | [AkiraOtok](https://github.com/AakiraOtok)        | <22520847@gm.uit.edu.vn> |



## Giới thiệu

Từ những kinh nghiệm thực tế trong việc áp dụng Machine Learning, tôi đã chứng kiến cuộc thi Home Credit - Credit Risk Model Stability trở thành một trong những thách thức phức tạp nhất mà tôi từng gặp. Dữ liệu cung cấp rất lớn và đa dạng, bao gồm thông tin từ nhiều nguồn khác nhau và phản ánh đời sống tài chính của các cá nhân. Để giải quyết bài toán này, ta cần phải có kiến thức vững chắc về lĩnh vực tài chính, khả năng phân tích sâu sắc và hiểu biết sâu về dữ liệu trong lĩnh vực ngân hàng.

Dựa trên sự hiểu biết của tôi, tôi nhận thấy có hai yếu tố cực kỳ quan trọng để xây dựng một mô hình xuất sắc cho cuộc thi này:

* Thực hiện Feature Engineering: Việc tạo ra những đặc trưng (features) tốt từ dữ liệu gốc là một yếu tố quan trọng đối với thành công của mô hình. Quá trình này đòi hỏi sự sáng tạo và khả năng hiểu biết sâu về tài chính, để tìm ra những thông tin quan trọng, loại bỏ nhiễu và chọn lọc những đặc trưng quan trọng nhất để đưa vào mô hình.
* Sử dụng sự kết hợp đa dạng của các mô hình Machine Learning: Thay vì dựa chỉ vào một mô hình duy nhất, tôi tin rằng sự kết hợp giữa nhiều mô hình khác nhau có thể mang lại kết quả tốt hơn. Bằng cách sử dụng các thuật toán và phương pháp khác nhau, ta có thể khai thác sự mạnh mẽ của từng mô hình và tận dụng đa dạng hóa trong việc dự đoán rủi ro tín dụng.

Với việc thực hiện kỹ lưỡng cả hai yếu tố này, tôi tin rằng ta có thể xây dựng một mô hình ổn định và hiệu quả cho cuộc thi Home Credit - Credit Risk Model Stability.

## Mục tiêu 
Mục tiêu của đồ án này này là thiết kế model dự đoán về khả năng vỡ nợ của khách hàng (default on loans) dựa vào dữ liệu nội bộ (của tổ chức) và bên ngoài của từng khách hàng
Metric của cuộc thi sử dụng gini stability metric:
<p align="center">
$gini = 2 * AUC − 1$
</p>
Và chuẩn đo cuối cùng là:
<p align="center">
$stabilitymetric=mean(gini) + 88.0 × min(0,a) − 0.5 × std(residuals)$
</p>

Với $a$ là hệ số góc của đường thẳng hồi quy được tìm dựa trên dự đoán của mô hình.

Về dữ liệu, dữ liệu có một sự đa dạng ổn định, phù hợp cho mục tiêu của thiết kế model, trong đó có sự đặc biệt ở tập dữ liệu đó là được phân chia dựa trên giá trị `depth`: 
- depth = 0. Những đặc trưng tĩnh với từng `case_id` (gender, age ...)
- depth = 1. Những đặc trưng liên quan tới các bản ghi lịch sử của từng `case_id`  (previous applications, loans ...) được đánh số bởi `num_group1`
- depth = 2. Những đặc trưng cho biết thông tin chi tiết hơn đối với một vài trưng ở depth = 1 được đánh số bởi `num_group1` và `num_group2`
  
## Feature Engineering

Trước khi bắt đầu với việc gộp dữ liệu thì chúng tôi có biến đổi cột theo đúng định dạng được chỉ định ở từng đặc trưng ví dụ actualdpd_943P, amount_1115A ...
- `[P, A]` sang `Float64`
- `[M]` sang `String`
- `[D]` hoặc `date_decision` sang ` Date` 
```python
def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df
```
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

#### Tạo các biểu thức tổng hợp
Khi bắt đầu gộp dữ liệu, với bộ dữ liệu là depth 1 hoặc 2, quá trình sẽ thực hiện sinh các đặc trưng tổng hợp như sau:

- Đặc trưng có hậu tố `P` và `A` :  Sinh các cột số với các biểu thức tổng hợp như giá trị lớn nhất (max), giá trị đầu tiên (first), và giá trị trung bình (mean).
- Đặc trưng có hậu tố ` M` : Sinh các cột chuỗi với biểu thức mode (giá trị xuất hiện nhiều nhất) và giá trị lớn nhất (max).
- Đặc trưng có hậu tố `D` : Sinh các cột ngày tháng với các biểu thức như giá trị lớn nhất (max), giá trị đầu tiên (first), và giá trị trung bình (mean).
- Các cột có hậu tố `T, L` hoặc chứa `num_group` : Sinh các cột với biểu thức tổng hợp như giá trị lớn nhất (max) và giá trị đầu tiên (first).

#### Xử lý gom nhóm

##### Gom nhóm các cột theo số lượng giá trị khuyết
- Tạo một DataFrame chỉ chứa các giá trị khuyết (NaN) của các cột không phải là 'category'.
- Nhóm các cột theo số lượng giá trị khuyết và lưu vào từ điển nans_groups.
```python
nans_df = df_train[nums].isna()
nans_groups = {}
for col in nums:
    cur_group = nans_df[col].sum()
    try:
        nans_groups[cur_group].append(col)
    except:
        nans_groups[cur_group] = [col]
del nans_df; x = gc.collect()
```
##### Gom nhóm theo độ tương quan của cát cột
- Nhóm các cột có tương quan cao hơn ngưỡng cho trước (threshold=0.8). Mỗi nhóm sẽ chứa các cột có mối tương quan lớn hơn hoặc bằng 0.8 với nhau.
```python
def group_columns_by_correlation(matrix, threshold=0.8):
    correlation_matrix = matrix.corr()
    groups = []
    remaining_cols = list(matrix.columns)
    while remaining_cols:
        col = remaining_cols.pop(0)
        group = [col]
        correlated_cols = [col]
        for c in remaining_cols:
            if correlation_matrix.loc[col, c] >= threshold:
                group.append(c)
                correlated_cols.append(c)
        groups.append(group)
        remaining_cols = [c for c in remaining_cols if c not in correlated_cols]
    return groups
```
##### Chọn các cột cần sử dụng
- Đối với mỗi nhóm cột có cùng số lượng giá trị khuyết, nếu nhóm có nhiều hơn một cột, sẽ tiếp tục nhóm các cột dựa trên mối tương quan.
  Sau đó, chọn cột đại diện cho mỗi nhóm và thêm vào danh sách uses.
  Nếu nhóm chỉ có một cột, thêm cột đó trực tiếp vào danh sách uses.
```python
uses = []
for k, v in nans_groups.items():
    if len(v) > 1:
        Vs = nans_groups[k]
        grps = group_columns_by_correlation(df_train[Vs], threshold=0.8)
        use = reduce_group(grps)
        uses = uses + use
    else:
        uses = uses + v
    print('####### NAN count =', k)
```
- Trong đó hàm reduce_grop(grps) nhận vào các nhóm cột và chọn một cột đại diện cho mỗi nhóm dựa trên số lượng giá trị duy nhất (nunique). Cột có số lượng giá trị duy nhất    lớn nhất sẽ được chọn làm đại diện cho nhóm.
```python
def reduce_group(grps):
    use = []
    for g in grps:
        mx = 0; vx = g[0]
        for gg in g:
            n = df_train[gg].nunique()
            if n > mx:
                mx = n
                vx = gg
        use.append(vx)
    print('Use these', use)
    return use
```

#### Xử lý tạo mới đặc trưng
- Sinh thêm đặc trưng era từ first_birth_259D để xác định phạm vi giai đoạn:
```python 
df_base = df_base.with_columns(((pl.col("first_birth_259D") / 10).floor() * 10).alias("era").cast(pl.Int32))
```

### Những phát hiện
-  Bộ data bị lệch rất lớn 
-  `refreshdate_3813885D ` từ các tệp credit_bureau_a1*. Giá trị tối thiểu của nó cho mỗi ID trường hợp thường luôn bằng 3/1/2019 . Một lý do có thể là nó đã được sử dụng để điền vào các giá trị NaN. `refreshdate_3813885D` và `date_decision`  có một tương quan gần như hoàn hảo.





## Base Model

Mô hình tốt nhất của chúng tôi dựa trên mô hình được công bố trên notebook : Home Credit (LGB + Cat ensemble).

Notebook sử dụng hai mô hình chính là **lgboost** và **catboost**. Sau đó sử dụng **VotingModel** để kết hợp các kết quả dự đoán. Các chi tiết được mô tả cụ thể bên dưới đây : 

### Về cách khởi tạo các mô hình

Đầu tiên, lược bỏ cột 'target', 'case_id' và 'week_num' do các thông tin này sẽ bị che đi trong tập test :

```python 
X = df_train.drop(columns=["target", "case_id", "week_num"])
y = df_train["target"]
```
Với việc phân nhóm, và bộ dữ liệu rất mất cân bằng nên nhóm chúng tôi sẽ sử dụng `StratifiedGroupKFold`
bởi vì `StratifiedGroupKFold` là một biến thể của k-fold cross-validation được thiết kế đặc biệt để xử lý các tình huống mà dữ liệu không chỉ mất cân bằng về các lớp (labels) mà còn có thể chứa các nhóm (groups) mà ta muốn đảm bảo rằng các nhóm này không bị phân chia vào các tập huấn luyện và kiểm tra cùng một lúc. 
```python
cv = StratifiedGroupKFold(n_splits=5, shuffle=False)
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
    eval_metric='AUC',
    task_type='GPU',
    learning_rate=0.03,
    iterations=n_est)
    clf.fit(train_pool, eval_set=val_pool, verbose=300)
    fitted_models_cat.append(clf)
```

Tương tự đối với lgb

```python
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set = [(X_valid, y_valid)],
        callbacks = [lgb.log_evaluation(200), lgb.early_stopping(100)] )
    fitted_models_lgb.append(model)
```

Hyper param cho lgb là : 

```python 
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "max_depth": 10,  
    "learning_rate": 0.05,
    "n_estimators": 2000,  
    "colsample_bytree": 0.8,
    "colsample_bynode": 0.8,
    "verbose": -1,
    "random_state": 42,
    "reg_alpha": 0.1,
    "reg_lambda": 10,
    "extra_trees":True,
    'num_leaves':64,
    "device": device, 
    "verbose": -1,
}
```

Sau khi huấn luyện mô hình, sử dụng một VotingModel để đưa ra dự đoán, VotingModel chỉ đơn giản là lấy mean của các dự đoán : 

```python 
def predict_proba(self, X):
        
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators[:5]]
        
        X[cat_cols] = X[cat_cols].astype("category")
        y_preds += [estimator.predict_proba(X) for estimator in self.estimators[5:]]
        
        return np.mean(y_preds, axis=0)
```

## Các lần thử

Ngoài notebook ở trên ra, chúng tôi có những thử nghiệm trên các note book khác, chúng tôi sẽ kí hiệu cho từng note book đó để tiện theo dõi : 
* Home Credit Risk Mode: utility scripts : $A$
* seed=3107 is what you need : $B$
* Home Credit - Credit Risk Model Stability : $C$
* Fork of Credit Risk Prediction with LightGBM and C : $D$
* catboost_lightgbm_ensemble e463ae : $E$
* Home Credit : AutoML more features : $F$
* Home Credit (LGB + Cat ensemble) : $G$
* Essemble(cat + lgbm) : $H$

Kí hiệu $T$ + ... ám chỉ sử dụng note book $T$ và áp dụng thay đổi khác. Có một số notebook kết quả gốc không thực sự ấn tượng nên qua thử nghiệm không được lấy làm gốc cải tiến thêm, các kết quả bên dưới là trên private test :

| What     | Result   | 
| -------- | -------- |
|$E$ + xgb(nestimator=100)|0.51233|
|$E$ + xgb(nestimator=2000)|MLE|
|$E$ + xgb(nestimator=1200)|MLE|
|$D$ + "class_weight" :"balanced", auto_class_weights='Balanced',|0.45401|
|$D$ + n_splits=10 + "random_state": 3107 + random_seed = 3107|0.50816|
|$H$ + thêm feature "birth_year" từ "first_birth_259D", lọc outlier week_num = 0, year = 356|0.46785|
