# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:12:46 2022

@author: Andrés Mejía
"""
# Caso c: 
# 𝑌 ≡ 𝐷𝑃, 𝑋0 ≡ 1, 𝑋1 ≡ 𝑁𝑎𝑑𝑖𝑟, 𝑋2 ≡ df/dt

## Importación de los datos
import pandas as pd
import numpy as np
import xlsxwriter

Data = pd.read_excel('DATOS.xlsx', sheet_name='Hoja1')
Data = round(Data,2)
## Principales Estadísticos
X1=Data.iloc[:,5]#DP_Potencia salida
X2=Data.iloc[:,1]# Nadir
X5=Data.iloc[:,2]#f_final
X4=Data.iloc[:,3]#V_final
X3=Data.iloc[:,4]#ROCOF
X6=Data.iloc[:,6]#Demanda

def status(x) : 
    return pd.Series([x.count(),x.min(),x.max(),x.mean(), x.median(), x.std(),x.var(),x.skew(),x.kurt(), np.abs(x.std()/x.mean())],
                     index=['No. datos','V. mínimo','V. máximo', 'Media','Mediana','Desv. estándar','Varianza',
                            'Coef. Asimetría','Coef. Kurtosis', 'Coef. Variación'])
result= pd.DataFrame(np.array([X1,X2,X3,X4,X5]).T, columns=['DP [MW]','Nadir [Hz]','RoCoF [Hz/s]','Vend [kV]','fend [Hz]'])
Estadisticos=result.apply(status)
print(Estadisticos)

datos= pd.DataFrame(np.array([X1,X2,X3,X4,X5]).T, 
                    columns=['DP [MW]','Nadir [Hz]','RoCoF [Hz/s]','Vend [kV]','fend [Hz]'])

## Matriz de correlación
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")
plt.style.use('seaborn')
corr = datos.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(10,10))

ax=plt.title("Diagrama de correlación")
sns.heatmap(corr, cmap='Spectral_r', mask=mask, square=False, annot=True, linewidth=1, cbar_kws={"shrink" : 1})
plt.savefig("Caso_c_Correlacion")

## Matriz de varianzas y covarianzas
cov_mat = datos.cov()
print(cov_mat)
mask = np.zeros_like(cov_mat, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
fig, ax = plt.subplots(figsize=(10,10))
ax=plt.title("Diagrama de varianzas y covarianzas")
sns.heatmap(cov_mat, cmap='Spectral_r', square=False, annot=True, linewidth=1, cbar_kws={"shrink" : 1})
plt.savefig("Caso_c_Covarianza")
## Análisis de funciones de distribución 

flierprops = dict(markerfacecolor='g', color='g', alpha=0.5)

n_cols = 2
n_rows = int(np.ceil(datos.shape[-1]*2 / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
for i, (col) in enumerate(list(datos.columns)):
    mean = datos[col].mean()
    median = datos[col].median()
    sns.histplot(datos[col], ax=axes.flatten()[2*i], kde=True )
    sns.boxplot(x=datos[col], orient='h', ax=axes.flatten()[2*i+1], color='g')
    axes.flatten()[2*i+1].vlines(mean, ymin = -1, ymax = 1, color='r', label=f"Variable: [{col}]\nMedia: {mean:.2}\nMediana: {median:.2}")
    axes.flatten()[2*i+1].legend()
  
plt.tight_layout()
plt.savefig("Caso_c_Histograma y cajas")
## corelación de variables
# Scatter Matrix - Red Wine
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="darkgrid")
sns.pairplot(datos,palette="ch:s=-.2,r=.6")
plt.savefig("Caso_c_Correlacion_dispersion")
plt.show()

# MODELO DE REGRESIÓN LINEAL MÚLTIPLE
## División del conjunto de datos
X= pd.DataFrame(np.array([X2,X3]).T, columns=['Nadir [Hz]','RoCoF [Hz/s]'])
Y= pd.DataFrame(np.array([X1]).T, columns=['DP [MW]'])
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 2, test_size=0.2)
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)


## Entrenamiento del modelo de regresión lineal múltiple
X_train=X_train.to_numpy()
Y_train=Y_train.to_numpy()

from scipy import stats
from numpy import linalg as LA
#Linear Regression
N = len(Y_train)
ones=np.ones((N,))
Xreg=np.insert(X_train, 0, ones, axis=1)
B = LA.inv(Xreg.transpose()@Xreg)@Xreg.transpose()@Y_train
Yreg = Xreg@B
Error = Y_train - Yreg
Ymed = Y_train.mean()
VE = sum((Yreg-Ymed)**2) #Square Sum SS Explained variability
VNE = sum(Error**2) #Square Sum SS Unexplained variability
VT = sum((Y_train-Ymed)**2) #Square Sum SS Total variability
# Multiple Correlation
R2 = VE/VT # R square
R = np.sqrt(R2) # Multiple R
if sum(np.shape(X_train))>N:
    df_ve = np.shape(X_train)[1] # degrees of freedom of ve
else:
    df_ve = 1
  
df_vne = N-1-df_ve # degrees of freedom of vne
df_vt = N-1 # degrees of freedom of vt

MS_ve = VE/df_ve #mean square of ve
MS_vne = VNE/df_vne #mean square of vne
MS_vt = VT/df_vt #mean square of vt

St_error = np.sqrt(MS_vne) #Std. error of the estimate
St_error = np.array([St_error ])
R2_adj = 1-MS_vne/MS_vt # Adjusted R square
      
# F-test
   
F = MS_ve/MS_vne # F-test for the null hypothesis
f_pvalue = 1-stats.f.cdf(F, df_ve, df_vne) #P-value of F (Funcion de distribucion acumulada)

# Significance of B
Seb = np.sqrt(MS_vne*np.diag(LA.inv(Xreg.transpose()@Xreg))) #Coefficient standard error
Seb_array=np.array([Seb])
Seb_array_T=Seb_array.T
t_stat = B/Seb_array_T # t Stat of coefficients
b_pvalue = ((1 - stats.t.cdf(t_stat,df=df_vne))*(t_stat>=0) + (stats.t.cdf(t_stat,df=df_vne))*(t_stat<0))*2 # p-value of coefficients

df1 = pd.DataFrame()
df1['Parámetros'] = ['Coeficiente de vcorrelación múltiple R','Coeficiente de determinación R2','Coeficiente de determinación ajustado R2','Error estimado estándar']
df1['Valores'] = [R,R2,R2_adj,np.round(St_error,6) ]

df2 = pd.DataFrame()
df2[' '] = ['B0','B1','B2']
df2['Coeficientes'] = B
df2['Error Std.'] = Seb_array_T
df2['Prueba t'] = t_stat
df2['Significancia de t'] = b_pvalue

df3 = pd.DataFrame()
df3[''] = ['VE','VNE','VT']
df3['Grados de libertad'] =[df_ve,df_vne,df_vt]
df3['Suma de cuadrados SC'] = [VE ,VNE ,VT ]
df3['Media cuadrática MC'] = [MS_ve,MS_vne,MS_vt]
df3['Prueba F'] = [F,0,0]
df3['Significancia de F'] = [f_pvalue,0,0]


## Resultados del Entrenamiento de la Regresión
print('Resultados del Entrenamiento de la Regresión')
print(df1)
print('')
### Coeficientes del modelo
print('Coeficientes del modelo')
print(df2)
print('')
### Indicadores del análisis de varianza (ANOVA)
print('Indicadores del análisis de varianza (ANOVA)')
print(df3)
print('')
print('R2 inicial',R2)

##↨ Predicción
y_pred = B[0]*np.ones(len(X_test))+B[1]*X_test.iloc[:,0]+B[2]*X_test.iloc[:,1]
Y_pred = y_pred.to_frame()
Y_pred.columns =['DP[MW] - Datos estimados']
Y_test.columns =['DP[MW] - Datos prueba']

nadir=X_test.iloc[:,0]
Nadir = nadir.to_frame()
x1 = Nadir.to_numpy()

rocof=X_test.iloc[:,1]
Rocof = rocof.to_frame()
x2 = Rocof.to_numpy()

x2 = x1
y1 = Y_pred.to_numpy()
y2 = Y_test.to_numpy()
R2=df1.iloc[1,1]

plt.scatter(x1, y2, s = 30, color='blue', marker='o',label = "DP[MW] - Datos de prueba")
plt.scatter(x1, y1, s = 50, color='red', marker='^', label = "DP[MW] - Datos estimados")
plt.text(58.7, 0.2, 'R2 = ', fontsize=13)
plt.text(58.8, 0.2, R2, fontsize=13)
plt.xlabel('Nadir [Hz]')
plt.ylabel('Potencia de salida DP [MW]')
plt.legend()
plt.savefig("Caso_c_Resultados")
plt.show()




with pd.ExcelWriter('Resultados_caso_c.xlsx', engine='xlsxwriter') as writer:
    Estadisticos.to_excel(writer, sheet_name='Estadisticos')
    corr.to_excel(writer, sheet_name='M. correlación')
    cov_mat.to_excel(writer, sheet_name='M. var. y cov')
    df1.to_excel(writer, sheet_name='df1 Parámetros')
    df2.to_excel(writer, sheet_name='df2 Coeficientes')
    df3.to_excel(writer, sheet_name='df3 ANOVA')


X_test = X_test.to_numpy()
Y_test=Y_test.to_numpy()
## Prueba del modelo de regresión lineal múltiple
N = len(Y_test)
ones=np.ones((N,))
Xreg=np.insert(X_test, 0, ones, axis=1)
B = LA.inv(Xreg.transpose()@Xreg)@Xreg.transpose()@Y_test
Yreg = Xreg@B
Error = Y_test - Yreg
Ymed = Y_test.mean()
VE = sum((Yreg-Ymed)**2) #Square Sum SS Explained variability
VNE = sum(Error**2) #Square Sum SS Unexplained variability
VT = sum((Y_test-Ymed)**2) #Square Sum SS Total variability
# Multiple Correlation
R2 = VE/VT # R square
R = np.sqrt(R2) # Multiple R

print('R2 final',R2)



import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
from matplotlib import cm
def main():
    # Generate Data...
    numdata = 100
    x = rocof
    y = nadir
    z = y1

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 20, 20
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx), 
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot
    #plt.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
    #plt.scatter(x, y, c=z)
    #plt.show()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z, color='red', zorder=0)
    ax.plot_surface(xx, yy, zz, zorder=10, alpha=0.7, cmap=cm.ocean)
    ax.set_xlabel('df/dt [Hz]')
    ax.set_ylabel('Nadir [Hz]')
    ax.set_zlabel('DP [MW]')
    plt.show()
    text = "filler"
    plt.savefig("Caso_c_Hiperplano")
    
def polyfit2d(x, y, z, order=4):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    #ij = itertools.product(range(order+1), range(order+1))
    ij = xy_powers(order)
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    #ij = itertools.product(range(order+1), range(order+1))
    ij = xy_powers(order)
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

def xy_powers(order):
    powers = itertools.product(range(order + 1), range(order + 1))
    return [tup for tup in powers if sum(tup) <= order]

main()


