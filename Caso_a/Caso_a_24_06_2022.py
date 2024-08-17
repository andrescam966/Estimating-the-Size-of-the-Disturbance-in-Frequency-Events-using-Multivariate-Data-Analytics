# -- coding: utf-8 --
"""
Created on Thu Jun 23 13:12:46 2022

@author: Andrés Mejía
"""
# Caso a: 
# 𝑌 ≡ 𝐷𝑃, 𝑋0 ≡ 1, 𝑋1 ≡ 𝑁𝑎𝑑𝑖𝑟, 𝑋2 ≡ df/dt, 𝑋3 ≡ 𝑉𝑒𝑛𝑑, 𝑋4 ≡ 𝑓𝑒𝑛𝑑

import os
import pandas as pd
import numpy as np
import xlsxwriter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy import stats
from numpy import linalg as LA
import phik
from phik.report import plot_correlation_matrix
from joblib import parallel_backend

# Crear la carpeta de salida si no existe
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Cargar datos
file_path = 'DATOS.xlsx'
sheet_name = 'Hoja1'

if os.path.exists(file_path):
    Data = pd.read_excel(file_path, sheet_name=sheet_name)
    Data = round(Data, 2)
else:
    raise FileNotFoundError(f"El archivo '{file_path}' no se encuentra en el directorio actual.")

# Selección de columnas
X1 = Data.iloc[:, 5]  # DP_Potencia salida
X2 = Data.iloc[:, 1]  # Nadir
X5 = Data.iloc[:, 2]  # f_final
X4 = Data.iloc[:, 3]  # V_final
X3 = Data.iloc[:, 4]  # ROCOF
X6 = Data.iloc[:, 6]  # Demanda

# Función de estadísticos
def status(x): 
    return pd.Series([x.count(), x.min(), x.max(), x.mean(), x.median(), x.std(), x.var(), x.skew(), x.kurt(), np.abs(x.std()/x.mean())],
                     index=['No. datos', 'V. mínimo', 'V. máximo', 'Media', 'Mediana', 'Desv. estándar', 'Varianza',
                            'Coef. Asimetría', 'Coef. Kurtosis', 'Coef. Variación'])

result = pd.DataFrame(np.array([X1, X2, X3, X4, X5]).T, columns=['DP [MW]', 'Nadir [Hz]', 'RoCoF [Hz/s]', 'Vend [kV]', 'fend [Hz]'])
Estadisticos = result.apply(status)
print(Estadisticos)

datos = pd.DataFrame(np.array([X1, X2, X3, X4, X5]).T, columns=['DP', 'Nadir', 'RoCoF', 'Vend', 'fend'])

# Deshabilitar el paralelismo de joblib temporalmente
with parallel_backend('loky', n_jobs=1):
    # Matriz de correlación Phi_k
    phik_matrix = datos.phik_matrix(interval_cols=['DP', 'Nadir', 'RoCoF', 'Vend', 'fend'])

# Visualización del mapa de calor de Phik
safe_filename = "Caso_a_Phi_k_Correlacion".replace(":", "_").replace("/", "_").replace("\\", "_")
plot_correlation_matrix(phik_matrix.values, x_labels=phik_matrix.columns, y_labels=phik_matrix.index, title="Mapa de Calor de Correlación Phi_k")
plt.savefig(os.path.join(output_folder, safe_filename + ".png"))

# Matriz de varianzas y covarianzas
cov_mat = datos.cov()
print(cov_mat)

mask = np.zeros_like(cov_mat, dtype=bool)
mask[np.triu_indices_from(mask)] = True

fig, ax = plt.subplots(figsize=(10, 10))
sns.set(font_scale=2.3)
res1 = sns.heatmap(cov_mat, cmap='Spectral_r', square=False, annot=True, linewidth=1, cbar_kws={"shrink": 1}, fmt='.1g')
res1.set_yticklabels(res1.get_ymajorticklabels(), fontsize=32)
res1.set_xticklabels(res1.get_xmajorticklabels(), fontsize=32)

safe_filename = "Caso_a_Covarianza".replace(":", "_").replace("/", "_").replace("\\", "_")
plt.savefig(os.path.join(output_folder, safe_filename + ".png"))

# Análisis de funciones de distribución
flierprops = dict(markerfacecolor='g', color='g', alpha=0.5)
n_cols = 2
n_rows = int(np.ceil(datos.shape[-1] * 2 / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

for i, col in enumerate(datos.columns):
    mean = datos[col].mean()
    median = datos[col].median()
    sns.histplot(datos[col], ax=axes.flatten()[2*i], kde=True, stat="density")
    sns.set(font_scale=1)
    sns.boxplot(x=datos[col], orient='h', ax=axes.flatten()[2*i+1], color='g')
    axes.flatten()[2*i+1].vlines(mean, ymin=-1, ymax=1, color='r', label=f"Variable: [{col}]\nMedia: {mean:.2}\nMediana: {median:.2}")
    axes.flatten()[2*i+1].legend()

plt.tight_layout()

safe_filename = "Caso_a_Histograma_y_cajas".replace(":", "_").replace("/", "_").replace("\\", "_")
plt.savefig(os.path.join(output_folder, safe_filename + ".png"))

# Correlación de variables (Scatter Matrix)
sns.set_theme(style="whitegrid")
sns.set(font_scale=1.8)
sns.pairplot(datos, palette="ch:s=-.2,r=.6")

safe_filename = "Caso_a_Correlacion_dispersion".replace(":", "_").replace("/", "_").replace("\\", "_")
plt.savefig(os.path.join(output_folder, safe_filename + ".png"))
plt.show()

# MODELO DE REGRESIÓN LINEAL MÚLTIPLE
# División del conjunto de datos
X = pd.DataFrame(np.array([X2, X3, X4, X5]).T, columns=['Nadir [Hz]', 'RoCoF [Hz/s]', 'V.final [kV]', 'f.final [Hz]'])
Y = pd.DataFrame(np.array([X1]).T, columns=['DP [MW]'])
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=4, test_size=0.2)

# Entrenamiento del modelo de regresión lineal múltiple
X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()

N = len(Y_train)
ones = np.ones((N,))
Xreg = np.insert(X_train, 0, ones, axis=1)
B = LA.inv(Xreg.T @ Xreg) @ Xreg.T @ Y_train
Yreg = Xreg @ B
Error = Y_train - Yreg
Ymed = Y_train.mean()
VE = np.sum((Yreg - Ymed)**2)
VNE = np.sum(Error**2)
VT = np.sum((Y_train - Ymed)**2)
R2 = VE / VT
R = np.sqrt(R2)
df_ve = X_train.shape[1] if sum(X_train.shape) > N else 1
df_vne = N - 1 - df_ve
df_vt = N - 1

MS_ve = VE / df_ve
MS_vne = VNE / df_vne
MS_vt = VT / df_vt
St_error = np.sqrt(MS_vne)
R2_adj = 1 - MS_vne / MS_vt
F = MS_ve / MS_vne
f_pvalue = 1 - stats.f.cdf(F, df_ve, df_vne)

Seb = np.sqrt(MS_vne * np.diag(LA.inv(Xreg.T @ Xreg)))
t_stat = B / Seb
b_pvalue = ((1 - stats.t.cdf(t_stat, df=df_vne)) * (t_stat >= 0) + (stats.t.cdf(t_stat, df=df_vne)) * (t_stat < 0)) * 2

df1 = pd.DataFrame({
    'Parámetros': ['Coeficiente de correlación múltiple R', 'Coeficiente de determinación R2', 'Coeficiente de determinación ajustado R2', 'Error estimado estándar'],
    'Valores': [R, R2, R2_adj, np.round(St_error, 6)]
})

df2 = pd.DataFrame({
    ' ': ['B0', 'B1', 'B2', 'B3', 'B4'],
    'Coeficientes': B.flatten(),
    'Error Std.': Seb,
    'Prueba t': t_stat.flatten(),
    'Significancia de t': b_pvalue.flatten()
})

df3 = pd.DataFrame({
    '': ['VE', 'VNE', 'VT'],
    'Grados de libertad': [df_ve, df_vne, df_vt],
    'Suma de cuadrados SC': [VE, VNE, VT],
    'Media cuadrática MC': [MS_ve, MS_vne, MS_vt],
    'Prueba F': [F, 0, 0],
    'Significancia de F': [f_pvalue, 0, 0]
})

# Resultados del Entrenamiento de la Regresión
print('Resultados del Entrenamiento de la Regresión')
print(df1)
print('')
print('Coeficientes del modelo')
print(df2)
print('')
print('Indicadores del análisis de varianza (ANOVA)')
print(df3)
print('')

Je = np.sum(Error**2)
Je_med = Je / len(Y_train)
print('R2 inicial', R2)
print('Je', Je)
print('Je_med', Je_med)

# Predicción de datos de prueba
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()
y_pred = B[0] * np.ones(len(X_test)) + B[1] * X_test[:, 0] + B[2] * X_test[:, 1] + B[3] * X_test[:, 2] + B[4] * X_test[:, 3]

# Gráficos de resultados
sns.set(font_scale=1.4)
plt.scatter(X_test[:, 0], Y_test, s=30, color='blue', marker='o', label="DP[MW] - prueba")
plt.scatter(X_test[:, 0], y_pred, s=50, color='red', marker='^', label="DP[MW] - estimados")
plt.text(58.7, 0.2, 'R2 = ', fontsize=16)
plt.text(58.85, 0.2, str(R2), fontsize=16)
plt.xlabel('Nadir [Hz]')
plt.ylabel('Potencia de salida DP [MW]')
plt.legend()

safe_filename = "Caso_a_Resultados".replace(":", "_").replace("/", "_").replace("\\", "_")
plt.savefig(os.path.join(output_folder, safe_filename + ".png"))
plt.show()

# Guardar resultados en Excel
safe_filename = "Resultados_caso_a.xlsx".replace(":", "_").replace("/", "_").replace("\\", "_")
with pd.ExcelWriter(os.path.join(output_folder, safe_filename), engine='xlsxwriter') as writer:
    Estadisticos.to_excel(writer, sheet_name='Estadisticos')
    phik_matrix.to_excel(writer, sheet_name='Phi_k Correlacion')
    cov_mat.to_excel(writer, sheet_name='M. var. y cov')
    df1.to_excel(writer, sheet_name='df1 Parámetros')
    df2.to_excel(writer, sheet_name='df2 Coeficientes')
    df3.to_excel(writer, sheet_name='df3 ANOVA')

# Prueba del modelo de regresión lineal múltiple
Xreg = np.insert(X_test, 0, np.ones(len(Y_test)), axis=1)
B = LA.inv(Xreg.T @ Xreg) @ Xreg.T @ Y_test
Yreg = Xreg @ B
Error = Y_test - Yreg
Ymed = Y_test.mean()
VE = np.sum((Yreg - Ymed)**2)
VNE = np.sum(Error**2)
VT = np.sum((Y_test - Ymed)**2)
R2 = VE / VT
R = np.sqrt(R2)

Jt = np.sum(Error**2)
Jt_med = Jt / len(Y_test)
print('R2 final', R2)
print('Jt', Jt)
print('Jt_med', Jt_med)