# 데이터 전처리
import numpy as np
import pandas as pd

# CatBoost 모델
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # 평가 지표

# 데이터 불러오기기
df = pd.read_csv('AI_Resume_Screening.csv')

# Skills 원-핫 인코딩
skills_dummies = (
    df['Skills'].fillna('')
      .str.replace(r'\s*,\s*', ',', regex=True)
      .str.get_dummies(sep=',')
      .rename(columns=lambda c: f"Skill_{c.replace(' ', '_')}")
)

# 범주형은 문자열 그대로 유지 (CatBoost가 처리)
cat_cols = [c for c in ['Education', 'Certifications', 'Job Role'] if c in df.columns]

# 수치형 데이터 자동 리스트
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if 'AI Score (0-100)' in num_cols:        # 타겟(AI Score)은 제외
    num_cols.remove('AI Score (0-100)')

# 최종 X, y
X = pd.concat([df[cat_cols], skills_dummies, df[num_cols]], axis=1)

if 'AI Score (0-100)' not in df.columns:
    raise KeyError("타겟 'AI Score (0-100)' 컬럼을 찾을 수 없습니다.")

y = df['AI Score (0-100)'].astype(float).clip(0, 100)  # .clip(0, 100) = 0보다 작으면 0으로, 100보다 크면 100으로 변환

# 결측치 처리(문자열은 '', 숫자는 0)
for c in cat_cols:
    X[c] = X[c].fillna('None').astype(str)
for c in X.columns.difference(cat_cols):
    X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)

print("데이터 전처리 완료")

# 학습/평가 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CatBoost에 전달할 범주형 index
cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

train_pool = Pool(X_train, y_train, cat_features=cat_idx)
test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)
# Pool = CatBoost 모델에 사용되는 데이터를 로딩하고 저장하는 클래스

# CatBoost 회귀 학습 (AI Score 예측)
print("CatBoost 모델 학습")
model = CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',iterations=2000, learning_rate=0.05,
    depth=6, random_seed=42, early_stopping_rounds=100, verbose=100
)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# 성능 평가
print("CatBoost 성능 평가")
pred = model.predict(test_pool)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae  = mean_absolute_error(y_test, pred)
r2   = r2_score(y_test, pred)
print(f"RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f}")