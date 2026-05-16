import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('balanced_dataSet_more_pollutants.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())


df['ln_GDPpc'] = df.groupby('ADM1_NAME')['ln_GDPpc'].transform(lambda x: x.fillna(x.median()))
df['ln_GDPpc_sq'] = df.groupby('ADM1_NAME')['ln_GDPpc_sq'].transform(lambda x: x.fillna(x.median()))


df['ln_sum_NTL'] = df['ln_sum_NTL'].replace(-np.inf, 0)
df['ln_sum_NTL_sq'] = df['ln_sum_NTL_sq'].replace(-np.inf, 0)


duplicates = df[df.duplicated(subset=['ADM1_NAME', 'ADM2_NAME', 'year'], keep=False)]
print("\nDuplicates based on ADM1_NAME, ADM2_NAME, year:")
print(duplicates)

Q1 = df['mean_PM25'].quantile(0.25)
Q3 = df['mean_PM25'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = df[(df['mean_PM25'] < lower_bound) | (df['mean_PM25'] > upper_bound)]
print(f"\nOutliers in mean_PM25: {len(outliers)} rows")



encoder = OneHotEncoder(sparse_output=False, drop='first')  
adm1_encoded = encoder.fit_transform(df[['ADM1_NAME']])
adm1_columns = encoder.get_feature_names_out(['ADM1_NAME'])
adm1_df = pd.DataFrame(adm1_encoded, columns=adm1_columns, index=df.index)



df = df.drop(['ADM2_NAME'], axis=1)


df = df.drop(['geometry', 'ID'], axis=1)


numerical_features = [col for col in df.columns if col not in ['ADM1_NAME', 'mean_PM25', 'year'] and not col.startswith('ADM1_')]


imputer = SimpleImputer(strategy='median')
df[numerical_features] = imputer.fit_transform(df[numerical_features])


scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])


df_scaled = pd.concat([df_scaled, adm1_df], axis=1)

df_scaled = df_scaled.drop(['ADM1_NAME'], axis=1)




corr_matrix = df_scaled.corr()


high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

print("\nHighly correlated pairs (>0.8):")
for pair in high_corr:
    print(f"{pair[0]} and {pair[1]}: {pair[2]:.2f}")


features_to_remove = ['ln_GDPpc_sq', 'ln_mean_NTL_sq', 'ln_sum_NTL_sq', 'ln_mean_precipitations_sq', 'ln_mean_temperature_sq', 'ln_relative_denseVeg_area_sq', 'ln_pop_density_sq', 'ln_mean_altitude_mean_temperature', 'ln_burned_area_mean_temperature', 'ln_mean_precipitations_relative_denseVeg_area', 'ln_burned_area_relative_denseVeg_area']

df_scaled = df_scaled.drop(features_to_remove, axis=1, errors='ignore')


target = 'mean_PM25'
features = [col for col in df_scaled.columns if col != target]

X = df_scaled[features]
y = df_scaled[target]

print(f"\nFinal dataset shape: {df_scaled.shape}")
print(f"Features: {len(features)}")
print(f"Target: {target}")


df_scaled.to_csv('cleaned_dataset.csv', index=False)

print("\nData preparation completed. Cleaned dataset saved as 'cleaned_dataset.csv'")