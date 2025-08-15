import numpy as np
import pandas as pd

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

print("전처리 완료")