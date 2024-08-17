# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 20:09:46 2021

@author: Usuario
"""

# INTRODUCTION
import numpy 
import random
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
# INTRODUCTION

import os
os.environ["PATCH"]=r"C:\Program Files\DIgSILENT\PowerFactory 2022"+os.environ["PATH"]
import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2022 SP2\Python\3.9")

# Import power factory
import powerfactory as pf
app=pf.GetApplication() #get app

# access and active the project
user = app.GetCurrentUser()
project = app.ActivateProject("Sistema_Galápagos_V14 ENERO_2015") #active the project
prj = app.GetActiveProject()
app.Show()#opcional

#functions

# ******************************************************  
# Busses dictinionary

busses=app.GetCalcRelevantObjects('*.ElmTerm')
bus_dict={}
for bus in busses:
    bus_dict[bus.loc_name]=bus

# Generator dictionary
gens=app.GetCalcRelevantObjects('*.ElmSym')
gen_dict={}
for gen in gens:
    gen_dict[gen.loc_name]=gen
    
wtss=app.GetCalcRelevantObjects('*.ElmGenstat')
wts_dict={}
for wts in wtss:
    wts_dict[wts.loc_name]=wts
# 
# Load dictionary
lods=app.GetCalcRelevantObjects('*.ElmLod')
lod_dict={}
for lod in lods:
    lod_dict[lod.loc_name]=lod
    
# 1. Generación de demanda aleatoria

#1.1 Datos de la carga

n_mont=5
x_carga=6.5
sigma_carga=0.5
n_bins=10
Load_M=numpy.random.normal(x_carga,sigma_carga,n_mont)
lod_dict['R_Load'].SetAttribute('plini',7) 
wts_dict['PV SMASTP 1500kW'].SetAttribute('pgini',0.5)
Carga_M=Load_M.tolist()
# fig1, ax1 =plt.subplots()

# ax1.hist(Load_M, bins=n_bins, density=True, histtype='step', 
             # stacked=True,color='r', linewidth=1.5,label='X1')

               
                  
data=pd.DataFrame(data=[['G8-HYU',510,7.2,0.00130,0.6,1.6],
                            ['G9-HYU',510,7.2,0.00160,0.6,1.6],
                            ['G10-HYU',510,7.2,0.001750,0.6,1.6],
                            ['G11-HYU',510,7.2,0.00168,0.6,1.6],
                            ['G12-HYU',510,7.2,0.00180,0.6,1.6]],
                            # ['WTG-01',510,7.2,0.00142,0.350,0.55],
                            # ['WTG-02',510,7.2,0.00110,0.325,0.65],
                            # ['Planta PV SC',500,7.2,0.00150,0.8,1.5]],
                      columns = ['No. Generador','a [USD/h]','b [USD/h]',
                                 'c [USD/h]','Pmin [MW]','Pmax [MW]'])
print('-----------------------------------------------------------------------------')
print('|Consumo de combustible [USD/h] y límites de potencia de generadores [MW]   |')
print('-----------------------------------------------------------------------------')
print(data)
    
    
a=data['a [USD/h]']
b=data['b [USD/h]']
c=data['c [USD/h]']
  
Capacidad_minima=data['Pmin [MW]']
Capacidad_maxima=data['Pmax [MW]']

# 1.2 Datos de eólicos
# a = forma, n = escala
def weib(x,n,a):
    return (a / n) * (x / n)**(a - 1) * numpy.exp(-(x / n)**a)
x = numpy.arange(1, n_mont)
w1 = weib(x, 10, 2.)*1.50
plt.plot(w1)
plt.show()

s1 = numpy.random.weibull(3, n_mont)
s2 = numpy.random.weibull(4, n_mont)
s1 = 0.6/max(s1)*s1
s2 = 1.1/max(s2)*s2 
oc=0.801 
    
i=0
# ******************************************************
Nadir=[]
Rocof=[]
P_out=[]
V_end=[]
f_end=[]
Carga_t=[]
Rand_out = ['G9-HYU','G10-HYU','G11-HYU','G12-HYU','WTG_1','WTG_2','WTG_3','PV SMASTP 1500kW']

def selectRandom(Rand_out):
        return random.choice(Rand_out)
    
for i in range(n_mont):  
    
    Carga=Load_M[i]-3*s1[i]-s2[i]+oc
    Demanda_Potencia=Carga-3*s1[i]-s2[i]+oc  
    Lambda=max(b)
    P = []
   
    while abs(Demanda_Potencia)>0.00001:
        multiplicador = (Lambda-b)/2
        P =numpy.divide(multiplicador,c)
        P=numpy.minimum(P,Capacidad_maxima)
        P=numpy.maximum(P,Capacidad_minima)
        Demanda_Potencia=Carga-numpy.sum(P)
        Lambda=Lambda+((Demanda_Potencia*2)/(numpy.sum(numpy.divide(1,c))))
    
    squaredP = numpy.multiply(P,P)
    Costo_Combustible= numpy.add(a,numpy.multiply(b,P)+numpy.multiply(c,squaredP))
    total_Costo_Combustible=numpy.sum(Costo_Combustible)    
    Unidades = ['G8_HYU','G9_HYU','G10_HYU','G11_HYU','G12_HYU']#,'WTG_01','WTG_02','Planta PV SC']
    Salida = DataFrame(data = Unidades,columns=['Unidad'])
    Salida = DataFrame({'Unidad':Unidades,'Potencia de despacho [MW]':round(P,3),})
    print('-----------------------------------------------------------------------------')
    print('|Despacho de unidades de generación                                          ')
    print('-----------------------------------------------------------------------------')
   
    eol1 = {'Unidad':'WTG_01','Potencia de despacho [MW]':s1[i]} 
    Salida = Salida.append(eol1, ignore_index=True)
    eol2 = {'Unidad':'WTG_02','Potencia de despacho [MW]':s1[i]} 
    Salida = Salida.append(eol2, ignore_index=True)
    eol3 = {'Unidad':'WTG_03','Potencia de despacho [MW]':s1[i]} 
    Salida = Salida.append(eol3, ignore_index=True)   
    phv1 = {'Unidad':'Planta PV SC','Potencia de despacho [MW]':s2[i]} 
    Salida = Salida.append(phv1, ignore_index=True)
    print(Salida)
    print('No. simulación Montecarlo:',i)
    gen_out=selectRandom(Rand_out)
 
    lod_dict['R_Load'].SetAttribute('plini',Load_M[i]) 
    gen_dict['G8-HYU'].SetAttribute('pgini',P[0])
    gen_dict['G9-HYU'].SetAttribute('pgini',P[1])
    gen_dict['G10-HYU'].SetAttribute('pgini',P[2])
    gen_dict['G11-HYU'].SetAttribute('pgini',P[3])
    gen_dict['G12-HYU'].SetAttribute('pgini',P[4])
    wts_dict['WTG_1'].SetAttribute('pgini',s1[i])
    wts_dict['WTG_2'].SetAttribute('pgini',s1[i])
    wts_dict['WTG_3'].SetAttribute('pgini',s1[i])
    wts_dict['PV SMASTP 1500kW'].SetAttribute('pgini',s2[i])
    
    #Caso1: Salida de Generador G10-HYU
    shc_folder=app.GetFromStudyCase('IntEvt')
    shc_folder.CreateObject('EvtSwitch','Evento1_Salida_G10-HYU')
    event=shc_folder.GetContents() # read events
    evtswch_g10=event[0]#número de evento
    evtswch_g10.time=0 # time
    evtswch_g10.i_switch=0 # to open the circuit breaker

    
    if gen_out == 'G9-HYU':
       evtswch_g10.p_target=gen_dict['G9-HYU']
       P=gen_dict['G9-HYU'].GetAttribute('pgini')
       P_out.append(P) 
    elif gen_out == 'G10-HYU':
      evtswch_g10.p_target=gen_dict['G10-HYU']
      P=gen_dict['G10-HYU'].GetAttribute('pgini')
      P_out.append(P)
    elif gen_out == 'G11-HYU':
       evtswch_g10.p_target=gen_dict['G11-HYU']
       P=gen_dict['G11-HYU'].GetAttribute('pgini')
       P_out.append(P)
    elif gen_out == 'G12-HYU':
       evtswch_g10.p_target=gen_dict['G12-HYU']
       P=gen_dict['G12-HYU'].GetAttribute('pgini')
       P_out.append(P)
    elif gen_out == 'WTG_1':
       evtswch_g10.p_target=wts_dict['WTG_1']
       P=wts_dict['WTG_1'].GetAttribute('pgini')
       P_out.append(P)
    elif gen_out == 'WTG_2':
       evtswch_g10.p_target=wts_dict['WTG_2']
       P=wts_dict['WTG_2'].GetAttribute('pgini')
       P_out.append(P)
    elif gen_out == 'WTG_3':
       evtswch_g10.p_target=wts_dict['WTG_3']
       P=wts_dict['WTG_3'].GetAttribute('pgini')
       P_out.append(P)
    elif gen_out == 'PV SMASTP 1500kW':
       evtswch_g10.p_target=wts_dict['PV SMASTP 1500kW']
       P=wts_dict['PV SMASTP 1500kW'].GetAttribute('pgini')
       P_out.append(P)
    
    
    
    df=pd.DataFrame(Salida)
    pg=df['Potencia de despacho [MW]'].sum()-P_out[i]
    Carga_t1=Load_M[i]+oc
   
    print('Carga del sistema ',Carga_t1) 
    print('Generador desconectado',gen_out, '=',P_out[i]) 
    print('Potencia generación en línea ',pg) 
    #Dinamic simulation
    app.ResetCalculation()
    Carga_t.append(Carga_t1)
    ##Add element to Results Folder AllCalculation
    elmres_fre_bus4=app.GetFromStudyCase("Evento1.ElmRes")
    
    
    ##Initial conditions
    ini=app.GetFromStudyCase('ComInc')
    ini.p_resvar=elmres_fre_bus4
    ini.Execute()
    
    ##simulation
    sim=app.GetFromStudyCase('ComSim')
    sim.tstop=sim.tstop
    sim.Execute()
    
    #Export result of dinamic simulation
    comres = app.GetFromStudyCase('ComRes')
    comres.iopt_csel=0#export all variables
    comres.iopt_tsel=0#exportation time of data
    comres.iopt_locn =2
    comres.ciopt_head=1
    comres.pResult=elmres_fre_bus4 # from yopu get the result of PF
    comres.iopt_exp=4
    comres.f_name=r'E:\0. Academic\2. EPN\19. Paper\Programa\Evento1.txt'
    comres.Execute()
    
    #In order no to generate events
    app.ResetCalculation()
    
    
    #ploted
    
    re_bus4=numpy.genfromtxt('Evento1.txt', skip_header=2, filling_values=9999 )
    
    #NADIR ************************************************************************
   
    f_min=numpy.amin(re_bus4[:,2])
    Nadir.append(f_min)
    n_min=int(numpy.floor(numpy.amin(numpy.where(re_bus4[:,2] == f_min))).astype(int))
    t_min=float(re_bus4[n_min,0])
    
    #ROCOF ************************************************************************
    
    w=numpy.where(re_bus4[:,2]<=60)# tiempo Y frecuencia mínima
    w1=numpy.floor(w)
    a=int(w1[0,0])
    
    t=0.5+re_bus4[a,0]# delta de tiempo para simulacion en segundo
    
    z=numpy.where(re_bus4[:,0]>t)# tiempo y frecuencia a t=500ms
    z1=numpy.floor(z)
    h=int(z1[0,0])
 
    df_dt= (re_bus4[h,2]-re_bus4[a,2])/(re_bus4[h,0]-re_bus4[a,0])#derivative
    Rocof.append(df_dt)
    
    # END VOLTAGE
    va_end=re_bus4[:,1]
    v_end=va_end[len(va_end)-1]
    V_end.append(v_end*13.8)
    
    # END FRECUENCY
    fa_end=re_bus4[:,2]
    F_end=fa_end[len(fa_end)-1]
    f_end.append(F_end)
    
    
    #Dinamic simulation
    app.ResetCalculation()
    
    for w in range(1):
       event=shc_folder.GetContents()
       evtswch_g10=event[0]
       evtswch_g10.Delete()
    
Lista=pd.DataFrame(numpy.array([Nadir, f_end, V_end, Rocof, P_out, Carga_t]).T)
Lista.columns=('Nadir','f_end', 'V_end','Rocof','P_out','Load_M')
Lista.to_excel(r"E:\0. Academic\2. EPN\19. Paper\Programa\Datos1_RN.xlsx")

   


