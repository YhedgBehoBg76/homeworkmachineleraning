import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
plt.style.use('ggplot')
sns.set_palette("viridis")

df = pd.read_csv("AmesHousing.csv")

df.info()

# Категориальные: NaN -> "None"
none_cols = ['Alley', 'Pool QC', 'Fence', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond']
df[none_cols] = df[none_cols].fillna("None")

# Числовые: NaN -> 0
zero_cols = ['Bsmt Full Bath', 'Garage Area', 'Garage Cars', 'Total Bsmt SF', 'BsmtFin SF 1', 'Bsmt Unf SF']
df[zero_cols] = df[zero_cols].fillna(0)

# Lot Frontage: Медиана по соседству
df['Lot Frontage'] = df.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))

# Создание новых признаков
df['House Age'] = df['Yr Sold'] - df['Year Built']
df['Years Since Remod'] = df['Yr Sold'] - df['Year Remod/Add']
df['House Age'] = df['House Age'].clip(lower=0)
df['Years Since Remod'] = df['Years Since Remod'].clip(lower=0)

target = 'SalePrice'
drop_cols = ['Order', 'PID', 'SalePrice']

numeric_features = df.drop(columns=drop_cols).select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = df.drop(columns=drop_cols).select_dtypes(include=['object']).columns.tolist()

iso = IsolationForest(contamination=0.01, random_state=42)
outliers = iso.fit_predict(df[['Gr Liv Area', 'SalePrice']])
df_clean = df[outliers == 1].copy()
df_outliers = df[outliers == -1].copy()

# Сравнение R2 до и после
def evaluate_model(data):
    X = pd.get_dummies(data[numeric_features + categorical_features], drop_first=True)
    X = X.fillna(X.median())
    y = data[target]
    reg = Ridge().fit(X, y)
    return r2_score(y, reg.predict(X))

r2_before = evaluate_model(df)
r2_after = evaluate_model(df_clean)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='None')), 
                          ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical_features)
    ])

ridge_pipe = Pipeline([('prep', preprocessor), ('model', Ridge(alpha=10.0))])
ridge_pipe.fit(df_clean.drop(columns=drop_cols), df_clean[target])

# Получаем топ-10 признаков
ohe_names = ridge_pipe.named_steps['prep'].transformers_[1][1].named_steps['ohe'].get_feature_names_out(categorical_features)
all_names = numeric_features + list(ohe_names)
importance = pd.Series(np.abs(ridge_pipe.named_steps['model'].coef_), index=all_names).sort_values(ascending=False).head(10)

X_seg = df_clean[numeric_features].fillna(0)
X_seg_scaled = StandardScaler().fit_transform(X_seg)
kmeans = KMeans(n_clusters=5, random_state=42)
df_clean['Segment'] = kmeans.fit_predict(X_seg_scaled)

pca_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=10)),
    ('model', Ridge())
])
pca_pipe.fit(df_clean[numeric_features], df_clean[target])
pca_r2 = r2_score(df_clean[target], pca_pipe.predict(df_clean[numeric_features]))

fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# График 1: Аномалии
sns.scatterplot(data=df_clean, x='Gr Liv Area', y='SalePrice', ax=axes[0, 0], alpha=0.5, label='Нормальные')
sns.scatterplot(data=df_outliers, x='Gr Liv Area', y='SalePrice', ax=axes[0, 0], color='red', label='Аномалии')
axes[0, 0].set_title(f'Удаление аномалий (R2 вырос: {r2_before:.3f} -> {r2_after:.3f})')

# График 2: Важность признаков
importance.plot(kind='barh', ax=axes[0, 1], color='skyblue')
axes[0, 1].set_title('Топ-10 самых важных признаков (Ridge)')

# График 3: Сезонность (Месяцы)
sns.lineplot(data=df_clean, x='Mo Sold', y='SalePrice', ax=axes[1, 0], estimator='median', marker='o', color='green')
axes[1, 0].set_title('Сезонность: Медианная цена по месяцам')
axes[1, 0].set_xticks(range(1, 13))

# График 4: Динамика цен (Года)
sns.barplot(data=df_clean, x='Yr Sold', y='SalePrice', ax=axes[1, 1], estimator='mean', palette='OrRd')
axes[1, 1].set_title('Динамика цен по годам (Кризис 2008)')

plt.tight_layout()
plt.show()

print(f"Качество регрессии на 10 компонентах PCA: {pca_r2:.4f}")
print(f"Средняя цена в 2007: ${df_clean[df_clean['Yr Sold'] == 2007][target].mean():,.0f}")
print(f"Средняя цена в 2009: ${df_clean[df_clean['Yr Sold'] == 2009][target].mean():,.0f}")

spring_price = df_clean[df_clean['Mo Sold'].isin([3, 4, 5])][target].median()
winter_price = df_clean[df_clean['Mo Sold'].isin([12, 1, 2])][target].median()
print(f"Медиана Весна: ${spring_price:,.0f} vs Зима: ${winter_price:,.0f}")
