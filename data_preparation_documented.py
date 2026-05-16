# =============================================================================
# PREPARACIÓN DE DATOS PARA ANÁLISIS DE CONTAMINACIÓN DEL AIRE EN COLOMBIA
# =============================================================================
#
# Este documento describe el proceso de preparación de datos siguiendo la metodología
# CRISP-ML(Q) para un estudio sobre contaminación del aire (PM2.5) en municipios colombianos.
# Está escrito de manera clara y sencilla para que cualquier persona pueda entenderlo,
# incluso sin conocimientos técnicos avanzados.
#
# Autores: 
# Fecha: 
# Dataset: balanced_dataSet_more_pollutants.csv
# =============================================================================

# =============================================================================
# 1. INTRODUCCIÓN
# =============================================================================
#
# Imagina que tienes un montón de datos sobre el aire que respiramos en diferentes
# ciudades de Colombia. Estos datos incluyen mediciones de contaminación, luces nocturnas,
# lluvia, temperatura y otras cosas que podrían afectar la calidad del aire.
#
# Pero los datos crudos no se pueden usar directamente en una computadora para hacer
# predicciones. Es como tener ingredientes para una receta, pero sin cocinarlos primero.
# La preparación de datos es ese proceso de "cocina" que hace que los datos estén listos
# para el análisis.
#
# En este documento, explicaremos paso a paso cómo limpiamos y preparamos estos datos
# para estudiar la contaminación del aire.

# =============================================================================
# 2. CARGA DE DATOS Y LIBRERÍAS
# =============================================================================
#
# Primero, necesitamos herramientas para trabajar con los datos. Usamos bibliotecas
# de Python que son como cajas de herramientas especializadas:
# - pandas: Para manejar tablas de datos (como Excel pero más poderoso)
# - numpy: Para cálculos matemáticos
# - sklearn: Para técnicas de aprendizaje automático
# - matplotlib y seaborn: Para hacer gráficos (aunque no los usamos mucho aquí)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Ahora cargamos nuestros datos. Es como abrir un libro lleno de información.
df = pd.read_csv('balanced_dataSet_more_pollutants.csv')

print("¡Datos cargados exitosamente!")
print(f"El dataset tiene {df.shape[0]} filas y {df.shape[1]} columnas.")
print("Cada fila representa un municipio colombiano en un año determinado.")

# =============================================================================
# 3. EXPLORACIÓN INICIAL DE LOS DATOS
# =============================================================================
#
# Antes de cocinar, necesitamos ver qué ingredientes tenemos. Vamos a explorar
# un poco nuestros datos para entenderlos mejor.

print("\n=== INFORMACIÓN GENERAL DEL DATASET ===")
print("Columnas disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print(f"\nTipos de datos en cada columna:")
print(df.dtypes)

print(f"\n¿Cuántos valores faltan en cada columna?")
missing_data = df.isnull().sum()
for col, missing in missing_data.items():
    if missing > 0:
        print(f"- {col}: {missing} valores faltantes")

# =============================================================================
# 4. LIMPIEZA DE DATOS - PRIMERA PARTE
# =============================================================================
#
# Los datos reales nunca son perfectos. A veces faltan valores, o hay errores.
# Esta es la parte de "limpiar la cocina" antes de cocinar.
#
# Problema 1: Valores faltantes en ln_GDPpc y ln_GDPpc_sq
# Solución: Usar el valor promedio del departamento correspondiente.
# ¿Por qué? Porque municipios del mismo departamento suelen tener economías similares.

print("\n=== LIMPIEZA DE VALORES FALTANTES ===")
print("Imputando valores faltantes de PIB per cápita por departamento...")

# Para ln_GDPpc (logaritmo del PIB per cápita)
df['ln_GDPpc'] = df.groupby('ADM1_NAME')['ln_GDPpc'].transform(
    lambda x: x.fillna(x.median())
)

# Para ln_GDPpc_sq (cuadrado del logaritmo del PIB)
df['ln_GDPpc_sq'] = df.groupby('ADM1_NAME')['ln_GDPpc_sq'].transform(
    lambda x: x.fillna(x.median())
)

print("✓ Valores de PIB imputados correctamente.")

# Problema 2: Valores infinitos negativos en columnas de luces nocturnas
# Solución: Reemplazar con 0, porque representa "sin luces" (áreas sin actividad humana)
print("\nReemplazando valores infinitos en luces nocturnas...")

df['ln_sum_NTL'] = df['ln_sum_NTL'].replace(-np.inf, 0)
df['ln_sum_NTL_sq'] = df['ln_sum_NTL_sq'].replace(-np.inf, 0)

print("✓ Valores infinitos corregidos.")

# =============================================================================
# 5. DETECCIÓN DE PROBLEMAS EN LOS DATOS
# =============================================================================
#
# Vamos a buscar otros problemas comunes en los datos.

# ¿Hay filas duplicadas?
print("\n=== VERIFICACIÓN DE DUPLICADOS ===")
duplicates = df[df.duplicated(subset=['ADM1_NAME', 'ADM2_NAME', 'year'], keep=False)]
if len(duplicates) == 0:
    print("✓ No hay filas duplicadas. Cada municipio aparece solo una vez por año.")
else:
    print(f"⚠ Encontradas {len(duplicates)} filas duplicadas.")

# ¿Hay valores demasiado extremos (outliers)?
print("\n=== DETECCIÓN DE VALORES EXTREMOS ===")
# Usamos el método del rango intercuartil (IQR) para detectar outliers
Q1 = df['mean_PM25'].quantile(0.25)  # Primer cuartil (25% de los datos)
Q3 = df['mean_PM25'].quantile(0.75)  # Tercer cuartil (75% de los datos)
IQR = Q3 - Q1  # Rango intercuartil

# Límites para considerar un valor como extremo
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['mean_PM25'] < lower_bound) | (df['mean_PM25'] > upper_bound)]
print(f"Encontrados {len(outliers)} valores extremos en la contaminación PM2.5.")
print("Estos podrían ser ciudades muy contaminadas o muy limpias.")
print("Los mantendremos porque representan situaciones reales.")

# =============================================================================
# 6. TRANSFORMACIÓN DE DATOS
# =============================================================================
#
# Ahora viene la parte creativa: transformar los datos para que la computadora
# los entienda mejor.

# Problema: Las computadoras no entienden texto como "Antioquia" o "Bogotá"
# Solución: Convertir texto en números usando "One-Hot Encoding"
print("\n=== CONVERSIÓN DE TEXTO A NÚMEROS ===")

# Creamos variables dummy para los departamentos
# Por ejemplo: es_Antioquia (1 si es Antioquia, 0 si no)
encoder = OneHotEncoder(sparse_output=False, drop='first')
adm1_encoded = encoder.fit_transform(df[['ADM1_NAME']])
adm1_columns = encoder.get_feature_names_out(['ADM1_NAME'])

adm1_df = pd.DataFrame(adm1_encoded, columns=adm1_columns, index=df.index)
print(f"✓ Creadas {len(adm1_columns)} columnas para representar los departamentos.")

# Eliminamos columnas que no necesitamos
print("\nEliminando columnas innecesarias...")
df = df.drop(['ADM2_NAME', 'geometry', 'ID', 'year'], axis=1)
print("✓ Columnas eliminadas: municipios específicos, geometría, ID y año.")

# =============================================================================
# 7. ESCALADO DE CARACTERÍSTICAS
# =============================================================================
#
# Imagina que tienes medidas en metros y kilómetros mezcladas. La computadora
# se confunde porque unas son grandes y otras pequeñas. El escalado hace que
# todas las medidas estén en la misma "escala".

print("\n=== ESCALADO DE DATOS NUMÉRICOS ===")

# Seleccionamos solo las columnas numéricas
numerical_features = [col for col in df.columns
                     if col not in ['ADM1_NAME', 'mean_PM25'] and
                     not col.startswith('ADM1_')]

print(f"Columnas numéricas a escalar: {len(numerical_features)}")
print("Ejemplos:", numerical_features[:5])

# Imputamos cualquier valor faltante restante con la mediana
imputer = SimpleImputer(strategy='median')
df[numerical_features] = imputer.fit_transform(df[numerical_features])

# Aplicamos escalado estándar (media = 0, desviación estándar = 1)
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])

print("✓ Datos escalados correctamente.")

# Agregamos las columnas de departamentos codificadas
df_scaled = pd.concat([df_scaled, adm1_df], axis=1)
df_scaled = df_scaled.drop(['ADM1_NAME'], axis=1)

# =============================================================================
# 8. SELECCIÓN DE CARACTERÍSTICAS
# =============================================================================
#
# No todas las variables son útiles. Algunas se repiten o no aportan nueva información.
# Es como elegir los mejores ingredientes para la receta.

print("\n=== ANÁLISIS DE CORRELACIÓN Y SELECCIÓN DE CARACTERÍSTICAS ===")

# Calculamos qué tan relacionadas están las variables entre sí
corr_matrix = df_scaled.corr()

# Buscamos parejas muy relacionadas (correlación > 0.8)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_value = abs(corr_matrix.iloc[i, j])
        if corr_value > 0.8:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_value
            ))

print(f"Encontradas {len(high_corr_pairs)} parejas altamente correlacionadas.")
print("Ejemplos de variables muy relacionadas:")
for pair in high_corr_pairs[:5]:
    print(".3f")

# Eliminamos variables redundantes según las recomendaciones
features_to_remove = [
    'ln_GDPpc_sq', 'ln_mean_NTL_sq', 'ln_sum_NTL_sq',
    'ln_mean_precipitations_sq', 'ln_mean_temperature_sq',
    'ln_relative_denseVeg_area_sq', 'ln_pop_density_sq',
    'ln_mean_altitude_mean_temperature', 'ln_burned_area_mean_temperature',
    'ln_mean_precipitations_relative_denseVeg_area',
    'ln_burned_area_relative_denseVeg_area'
]

df_scaled = df_scaled.drop(features_to_remove, axis=1, errors='ignore')
print(f"✓ Eliminadas {len(features_to_remove)} variables redundantes.")

# =============================================================================
# 9. RESULTADOS FINALES
# =============================================================================
#
# ¡Listo! Nuestros datos están preparados. Vamos a ver el resultado final.

print("\n" + "="*60)
print("RESUMEN FINAL DE LA PREPARACIÓN DE DATOS")
print("="*60)

# Definimos qué es lo que queremos predecir y qué datos usaremos
target = 'mean_PM25'  # Nuestra variable objetivo: contaminación PM2.5
features = [col for col in df_scaled.columns if col != target]

print(f" Tamaño final del dataset: {df_scaled.shape[0]} filas × {df_scaled.shape[1]} columnas")
print(f" Variable a predecir: {target} (concentración de PM2.5)")
print(f" Número de características: {len(features)}")
print(f"  Características principales:")
for i, feat in enumerate(features[:10], 1):
    print(f"   {i}. {feat}")
if len(features) > 10:
    print(f"   ... y {len(features) - 10} más")

# Guardamos el dataset limpio
output_filename = 'cleaned_dataset.csv'
df_scaled.to_csv(output_filename, index=False)
print(f"\n💾 Dataset guardado como: {output_filename}")

# =============================================================================
# 10. CONCLUSIONES
# =============================================================================
#
# Hemos transformado datos crudos en un conjunto de datos listo para el análisis.
# Este proceso es crucial porque:
#
# 1. **Datos limpios**: Eliminamos errores y valores faltantes
# 2. **Formato correcto**: Convertimos texto en números que las computadoras entienden
# 3. **Escala apropiada**: Todas las variables están en la misma escala
# 4. **Sin redundancias**: Eliminamos variables que no aportan nueva información
#
# Ahora podemos usar estos datos para crear modelos que predigan la contaminación
# del aire y ayudar a mejorar la calidad del aire en Colombia.
#
# ¡El siguiente paso es entrenar modelos de aprendizaje automático!

print("\n" + "="*60)
print("¡PREPARACIÓN DE DATOS COMPLETADA EXITOSAMENTE!")
print("Los datos están listos para la fase de modelado.")
print("="*60)

# =============================================================================
# REFERENCIAS Y NOTAS TÉCNICAS
# =============================================================================
#
# Este código sigue las mejores prácticas de la metodología CRISP-ML(Q):
# - CRoss Industry Standard Process for Data Mining
# - Adaptado para problemas de Machine Learning
#
# Técnicas utilizadas:
# - Imputación por mediana para valores faltantes
# - One-Hot Encoding para variables categóricas
# - Escalado Z-score (estandarización)
# - Análisis de correlación para selección de características
#
# Dataset original: https://data.mendeley.com/datasets/kd6tjgy9y8/1
# Contiene datos geoespaciales de municipios colombianos (2016)
#
# =============================================================================