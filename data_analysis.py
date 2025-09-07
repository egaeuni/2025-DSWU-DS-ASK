# 데이터 전처리
import numpy as np
import pandas as pd

# CatBoost 모델
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score   # 평가 지표

# 상관 분석
from scipy.stats import spearmanr

# 시각화
import shap
import matplotlib.pyplot as plt

# 데이터 불러오기
# 데이터 불러오기
df = pd.read_csv('/content/AI_Resume_Screening.csv')

# Skills 원-핫 인코딩 -> 하나의 셀에 여러 값이 들어갔기 때문에 분리
skills_dummies = (
    df['Skills'].fillna('')
      .str.replace(r'\s*,\s*', ',', regex=True)   # 공백 제거
      .str.get_dummies(sep=',')                   # 다중 값 → 여러 컬럼
      .rename(columns=lambda c: f"Skill_{c.replace(' ', '_')}")
)

# Skills 컬럼 제거 후, 스킬 원-핫 인코딩 컬럼들 추가
df = pd.concat([df.drop(columns=['Skills']), skills_dummies], axis=1)

# Recruiter Decision은 인코딩에서 제외
exclude_col = ['Recruiter Decision']
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.difference(exclude_col)

#그 외 범주형을 원-핫 인코딩 -> 하나의 샘플이 단일 값을 가지기 때문에 분리 필요 X
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# 최종 X, y
X = df_encoded.drop(columns=["AI Score (0-100)", 'Recruiter Decision'])
y = df_encoded["AI Score (0-100)"].astype(float).clip(0, 100)

print("전처리 완료")

# 학습/평가 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train)
test_pool  = Pool(X_test,  y_test)
# Pool = CatBoost 모델에 사용되는 데이터를 로딩하고 저장하는 클래스

# CatBoost 회귀 학습 (AI Score 예측)
print("[CatBoost 모델 학습]")
model = CatBoostRegressor(loss_function='RMSE',eval_metric='RMSE',iterations=2000, learning_rate=0.05,
    depth=6, random_seed=42, early_stopping_rounds=100, verbose=100
)
model.fit(train_pool, eval_set=test_pool, use_best_model=True)

# 성능 평가
print("\n[CatBoost 성능 평가]")
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
print("[SHAP 기준 상위 15개]")
print(imp_df.head(15))

## 방향성 추정 (스피어만 상관계수: +1에 가까울수록 정비례, -1에 가까울수록 반비례, 0에 가까우면 순위 상관 거의 X)
dir_sign = []

for i, col in enumerate(features):
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

print("\n[스피어만 상관계수를 기준으로 방향성까지 고려한 상위 15개]")
print(summary_df.head(15))

# 카테고리별 평균 SHAP 값 (각 카테고리 값이 AI Score를 얼마나 올리거나 내리는지)
# 범주형 데이터만 실행
def category_mean_shap(col, top_k=10):
    if col not in X_test.columns:
        raise KeyError(f"{col}은 X에 없는 컬럼입니다.")
    j = features.index(col)
    tmp = pd.DataFrame({'val': X_test[col], 'shap': phi[:, j]})
    return tmp.groupby('val', as_index=False)['shap'].mean()

for c in X_test.columns:
  print(f"\n[{c}의 평균 SHAP]")
  print(category_mean_shap(c))

# 특정 후보의 feature을 바꾸면 예측 점수가 어떻게 변할지 시뮬레이션
def simulate_changes(row, changes: dict):
    x_new = row.copy() # raw: X의 한 행 (= pd.Series)
    for k, v in changes.items(): # changes: 딕셔너리에 있는 값
        if k not in X.columns:
            raise KeyError(f"'{k}'는 X에 없는 컬럼입니다.")
        x_new[k] = v
    # 단일 행 Pool 구성
    pool = Pool(pd.DataFrame([x_new.values], columns=X.columns))
    return float(model.predict(pool)[0])

print("\n[후보 특성 변경 시뮬레이션]")

# AI Score가 가장 낮은 후보 선택
min_score_idx = y_test.idxmin()        # y_test에서 최소값 index
sample_row = X_test.loc[min_score_idx] # 해당 후보 샘플
base_pred = y_test.loc[min_score_idx]

# WHAT-IF 적용
modifications = {}
if 'Experience (Years)' in X.columns:
    modifications['Experience (Years)'] = 4  # 경력 설정
if 'Projects Count' in X.columns:
    modifications['Projects Count'] = 5  # 프로젝트 수 설정
### 다른 후보도 적용 가능 (자격증, 학력 등등...)

new_pred = simulate_changes(sample_row, modifications)

# 변경 전후 값 확인
for k, v in modifications.items():
    print(f"{k}: 원래 값 = {sample_row[k]} → 변경 값 = {v}")

print(f"원래 예측 점수 = {base_pred:.2f} → 변환 후 예측 점수 = {new_pred:.2f}")
print(f"점수 변화 Δ={new_pred - base_pred:+.2f}")

# SHAP TreeExplainer 생성
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 전역 SHAP 시각화
# Summary Dot Plot
shap.summary_plot(shap_values, X_test, plot_type="dot", max_display=10)

# Summary Bar Plot
shap.summary_plot(shap_values, X_test, plot_type="bar", max_display=10)

# 지역 SHAP 시각화 (대표 후보)
# AI Score 최소 후보 기준
min_idx = y_test.idxmin()
sample_row = X_test.loc[min_idx]

cat_cols = {
    'Skills': [c for c in X_test.columns if c.startswith('Skill_')],
    'Certifications': ['Certifications_AWS Certified',
                       'Certifications_Deep Learning Specialization',
                       'Certifications_Google ML']
}

# Certification 레이블 줄 바꿈 처리
rename_certi = {
    "Deep Learning Specialization": "Deep Learning\nSpecialization",
    "Google ML": "Google\nML",
    "AWS Certified": "AWS\nCertified"
}

# Skill 레이블 줄바꿈 처리
rename_skill = {
    "Deep_Learning" : "Deep\nLearning",
    "Machine_Learning" : "Machine\nLearning"
}

# 범주형 변수별 평균 SHAP 시각화
fig, axes = plt.subplots(
    1, 2, figsize=(12,5),
    gridspec_kw={"width_ratios":[2.5, 1.5]}  # Skills:Certifications = 3:1 비율
)

for i, (cat_name, cols) in enumerate(cat_cols.items()):
    mean_shap = {col: shap_values[X_test.columns.get_loc(col)].mean() for col in cols}
    mean_shap = pd.Series(mean_shap).sort_values(ascending=False)

    if cat_name == 'Skills':
      mean_shap = mean_shap.head(9)
      labels = [rename_skill.get(col.replace('Skill_', ''), col.replace('Skill_', '')) for col in mean_shap.index]
      mean_shap.plot(kind='bar', width=0.6, ax=axes[0])

    else:  # Certifications
      labels = [rename_certi.get(col.replace('Certifications_', '')) for col in mean_shap.index]
      mean_shap.plot(kind='bar', width=0.45, ax=axes[1])      

    axes[i].set_ylabel("Mean SHAP")
    axes[i].set_xticklabels(labels, rotation=rotation, ha=ha)
    axes[i].set_title(f"Mean SHAP of {cat_name}")

plt.tight_layout()
plt.show()

