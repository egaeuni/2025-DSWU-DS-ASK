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
print("\n[SHAP 기준 상위 15개]")
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

print("\n[스피어만 상관계수를 기준으로 방향성까지 고려한 상위 15개]")
print(summary_df.head(15))

# 카테고리별 평균 SHAP 값 (각 카테고리 값이 AI Score를 얼마나 올리거나 내리는지)
# 범주형 데이터만 실행
def category_mean_shap(col, top_k=10):
    if col not in cat_cols:
        raise ValueError(f"{col}은(는) 범주형이 아닙니다.")
    j = features.index(col)
    tmp = pd.DataFrame({'val': X_test[col].astype(str), 'shap': phi[:, j]})
    result = tmp.groupby('val', as_index=False)['shap'].mean()  # reset_index 효과
    return result.sort_values('shap', ascending=False).head(top_k)

for c in cat_cols:
    print(f"\n[{c}의 평균 SHAP]")
    print(category_mean_shap(c, top_k=10))

# 특정 후보의 feature을 바꾸면 예측 점수가 어떻게 변할지 시뮬레이션
def simulate_changes(row, changes: dict):
    x_new = row.copy() # raw: X의 한 행 (= pd.Series)
    for k, v in changes.items(): # changes: 딕셔너리에 있는 값
        if k not in X.columns:
            raise KeyError(f"'{k}'는 X에 없는 컬럼입니다.")
        x_new[k] = v
    # 단일 행 Pool 구성
    pool = Pool(pd.DataFrame([x_new.values], columns=X.columns), cat_features=cat_idx)
    return float(model.predict(pool)[0])

print("\n[후보 특성 변경 시뮬레이션]")

# AI Score가 가장 낮은 후보 선택
min_score_idx = y_test.idxmin()        # y_test에서 최소값 index
sample_row = X_test.loc[min_score_idx] # 해당 후보 샘플
base_pred = y_test.loc[min_score_idx]

# WHAT-IF 적용
modifications = {}
if 'Experience (Years)' in X.columns:
    modifications['Experience (Years)'] = 0  # 경력 설정
if 'Projects Count' in X.columns:
    modifications['Projects Count'] = 10  # 프로젝트 수 설정
## 다른 후보도 적용 가능 (자격증, 학력 등등...)

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

# Certifications 레이블 줄바꿈 처리
rename_certi = {
    "Deep Learning Specialization": "Deep Learning\nSpecialization",
    "Google ML": "Google\nML",
    "AWS Certified": "AWS\nCertified"
}

# 범주형 변수별 평균 SHAP 시각화
cat_cols = ['Education', 'Certifications']

fig, axes = plt.subplots(1, len(cat_cols), figsize=(12, 4))  # 1행 2열 (가로 배치)

for i, c in enumerate(cat_cols):
    j = X_test.columns.get_loc(c)
    tmp = pd.DataFrame({'category': X_test[c].astype(str), 'shap': shap_values[:, j]})
    mean_shap = tmp.groupby('category')['shap'].mean().sort_values(ascending=False)

    labels = [rename_certi.get(cat, cat) for cat in mean_shap.index]

    mean_shap.plot(kind='bar', ax=axes[i], width=0.6)

    axes[i].set_title(f"Mean SHAP of {c}")
    axes[i].set_ylabel("Mean SHAP")

    # 조건에 따라 xticks 회전 조정
    axes[i].set_title(f"Mean SHAP of {c}")
    axes[i].set_ylabel("Mean SHAP")
    axes[i].set_xticks(range(len(labels)))
    axes[i].set_xticklabels(labels, rotation=0, ha='center')

    print(f"\n[{c} 평균 SHAP 테이블]")
    print(mean_shap)

plt.tight_layout()
plt.show()


