#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat November  1 20:56:43 2020

@author: EJVM
"""
import numpy as np

######
## Modelo para implementar el backpropagation que para fines ilustrativos
## Tiene dos capas, 1 capa oculta y 1 capa de salida
## Cada capa tiene una sola neurona
## Las entradas de cada solo contienen una entrada y el bias


def forward_nn(w11, w12, w21, w22, x):
    h1 = neuron(w11, w12, x)
    o1 = neuron(w21, w22, h1)
    return o1

def sq_error(o1, yy):
    e = o1 - yy
    return e*e

def error_nn(w11, w12, w21, w22, x, yy):
    o1 = forward_nn(w11, w12, w21, w22, x)
    e = o1 - yy
    return e*e

def neuron(w11, w12, x):
    #TODO: activation function
    return w11*x + w12


# Derivada Parcial
# El primer argumento es la funcion a la que se le obtendrá la derivada
# El segundo argumento es el indice de la entrada de argumentos respecto al cual se hara la derivada
# El tercer argumento son los argumentos variables-pueden haber muchos. La funcion se invoca con esos valores
def diffp(fn, idx, *args):
    delta = 0.000000000001 
    
    # Se invoca la funcion con los argumentos como se ingresan
    y = fn(*args)
    
    # Ahora, se incrementa una de las entradas que corresponden al indice idx
    args = list(args)
    args[idx] += delta
    
    #Ahora, se calcula la salida despues de cambiar una de las entradas un poquito con el delta
    y1 = fn(*args)

    #Se regresa la razon de cambio calculada
    return (y1 - y)/delta



# Inicializar Variables
# Se generan 100 valores aleatorios para la variable X entre 0 y 1
X = np.random.random((100, 1))

# Se generan los valores correspondientes a 'y' correspondientes a una linea con pendiente de 3, intercepta al eje
# 'y' en cuatro y se le agrega un ruido aleatorio
y = 4 + 3 * X + .3*np.random.randn(100, 1)

# Los cuatro pesos de la red se inicializan en un valor
w11 = 0.5 #Peso asociado al valor Xi
w12 = 0.5 #Peso asociado al sesgo 1
w21 = 0.5 #Peso asociado al valor h1
w22 = 0.5 #Peso asociado al sesgo 2




# Razon de aprendizaje
eta = 0.01

# Se seleccionan 50 epocas
for epoch in range(50):
    
    # Cada patron en el vector X se itera para entrenar el modelo
    for i in range(len(X)):
        # Se recupera el i-esimo patron de entrada y de salida
        x = X[i]
        yy = y[i]
        
        if i % 50 == 0:
            print(epoch*len(X)+i, error_nn(w11, w12, w21, w22, x, yy))
            print("w11",w11,"W12",w12,"w21",w21,"W22",w22)    
        
        #Forward Pass
        # Calculo del valor de la neurona h1
        # w11 peso de Xi, w12 peso del bias 1
        h1 = neuron(w11, w12, x)
        # Calculo del valor de la neurona de salida 01
        # w21 peso de h1, w22 peso del bias 2
        o1 = neuron(w21, w22, h1)

        ## CAlcular la razon de cambio del Error con respecto a 01  
        dE_do1 = diffp(sq_error, 0, o1, yy)
        ## CAlcular la razon de cambio de O1 con respecto a w21
        do1_dw21 = diffp(neuron, 0, w21, w22, h1)
        ## Calcular la razon de cambio del Error con respecto a w21: con la regla de la cadena
        dE_w21 = dE_do1 * do1_dw21
        
        
        ## Calcular la razon de cambio de O1 con respecto a W22
        do1_dw22 = diffp(neuron, 1, w21, w22, h1)
        ## Calcular la razon de cambio de Error con respecto a W22: con la regla de la cadena
        dE_w22 = dE_do1 * do1_dw22
        
        
        do1_dh = diffp(neuron, 2, w21, w22, h1)
        dE_dh = dE_do1 * do1_dh
        dh_dw11 = diffp(neuron, 0, w11, w12, x)
        dE_dw11 = dE_dh * dh_dw11

        dh_dw12 = diffp(neuron, 1, w11, w12, x)
        dE_dw12 = dE_do1 * do1_dh * dh_dw12
        
        ## Calculamos que tanto los pesos afectan al error obtenido
        
        #Backpropagation
        ## Actualizamos los pesos con la razon de aprendizaje
        w11 = w11 - eta * dE_dw11
        w12 = w12 - eta * dE_dw12

        w21 = w21 - eta * dE_w21
        w22 = w22 - eta * dE_w22

        
h1 = neuron(w11, w12, X)
ycap1 = neuron(w21, w22, h1)

from matplotlib import pyplot as plt

plt.figure(figsize=(11,4))
# plt.subplot()
# plt.scatter(X, ycap, color='r')
plt.subplot()
plt.plot(X, ycap1, color='g')
plt.subplot()
plt.scatter(X, y, color='b')
plt.show()

print("w11",w11,"W12",w12,"w21",w21,"W22",w22)

