# 데이터 전처리
import numpy as np
import pandas as pd

# CatBoost 모델
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # 평가 지표

# 상관 분석
from scipy.stats import spearmanr

# 데이터 불러오기
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

# CatBoost 모델 전역 해석
shap_vals = model.get_feature_importance(test_pool, type='ShapValues')  # CatBoost 내부의 SHAP
phi = shap_vals[:, :-1]  # 마지막 열(base value) 제거 
features = X_test.columns.tolist()

# 평균 절대 SHAP으로 전역 중요도 = 어떤 Feature가 전반적으로 중요한지
mean_abs_shap = np.abs(phi).mean(axis=0) # 각 feature가 전체 예측에 끼친 영향의 평균 절댓값 (크면 중요)
imp_df = pd.DataFrame({'feature': features, 'mean_abs_shap': mean_abs_shap})
imp_df = imp_df.sort_values('mean_abs_shap', ascending=False)
print("\nSHAP 기준 상위 15개")
print(imp_df.head(15))

# # 방향성 추정 (스피어만 상관계수: +1에 가까울수록 정비례, -1에 가까울수록 반비례, 0에 가까우면 순위 상관 거의 X)
dir_sign = []

for i, col in enumerate(features):
    # 범주형 컬럼 문자열이기 때문에 코드로 임시 변환
    if col in cat_cols:
        vals = pd.Categorical(X_test[col], categories=sorted(X_test[col].unique())).codes
    else:
        vals = X_test[col].values
    # 상관 계산
    if np.all(vals == vals[0]):
        dir_sign.append(np.nan)
        # 모든 값이 같으면 상관계수를 계산할 수 없으므로 Nan 값
    else:
        rho, _ = spearmanr(vals, phi[:, i])
        dir_sign.append(rho)
        # 값이 모두 같지 않으면 정상적으로 스피어만 상관계수 계산
        # rho -> 특성값(vals)과 SHAP 값(phi[:, i]) 간의 순위 상관계수

dir_df = pd.DataFrame({'feature': features, 'spearman(val, shap)': dir_sign})
summary_df = imp_df.merge(dir_df, on='feature', how='left')

print("\n스피어만 상관계수를 기준으로 방향성까지 고려한 상위 15개")
print(summary_df.head(15))