# PARTCILE FILTER

Este proyecto trata de aplicar un filtro de partículas para poder determinar los valores del estado oculto de un sistema dinámico a partir de observaciones ruidosas.

En general, se asume un sistema de la siguiente forma:

- Ecuación de estado

$$
x_{t}\; =\; f(X_{t-1}) + w_t, \;\;\; w_t \sim N(0,Q)
$$

- Ecuacion observacion

$$
y_t = H x_t + v_t, \;\;\; v_t \sim N(0,R)
$$

La estructura del proyecto es la siguiente

```
PARTICLE_FILTER/
│
│
├──class_particle_filter.py # Script con la clase ParticleFilter
│                             y sus subclases BootstrapParticleFilter │                             y AuxiliaryParticleFilter
│
├──plot.py                  # Script con funciones para plotear gráficas
│
├──simulations.py           # Script para simular sistemas 
│
├──utilities.py             # Script con funciones auxiliares para  
│                             ParticleFilter
│
├──demos                    # Carpetas con notebooks con demostraciones
```

- Script [class_particle_filter.py]:

    - Clase [ParticleFilter]: la clase base del proyecto para realizar un filtro de partículas

    - Clase [BootstrapParticleFilter]: subclase de ParticleFilter. Toma como distribución de importancia la distribución a priori $p(\phi_{t} \mid \phi_{t-1})$

    - Clase [AuxiliaryParticleFilter]: subclase de ParticleFilter. Utiliza la observación $y_t$ para identificar cuáles de las partículas en $t-1$ tienen más probabilidades de generar estados compatibles con la observación. En este caso en concreto, se toma como proposal la distribución a priori.

- Script [plot.py]:

    - Función [plot_results]: función graficar los resultados del filtrado.

    - Función [plot_sorted_weights]: función para graficar los pesos de las partículas (ordenadas según su peso). 

- Script [simulations.py]: 

    - Función [system_simulation]: a partir de la definición de las ecuaciones de estado y de observación, simula el sistema para una longitud de tiempo dado y devuelve dos arrays con los estados y las observaciones en cada tiempo.

- Script [utilities.py]: este script contiene dos funciones para manejar las dimensiones de las matrices que describen el sistema. Esto es especialmente útil al usar la función *random.multivariate_normal* de *numpy* para tomar muestras de una distribución multivariante normal. 
    
    - Función [as_vector]: asegura que el vector estado sea de la forma *(state_dim, )*

    - Función [as_covariance_matrix]: asegura que las matrices de covarianza sean de la forma *(state_dim, state_dim)* y *(observation_dim, observation_dim)*.

La carpeta [demos] se tienen los siguientes ejemplos:

- apf_lineal_examples: aplicacion del Auxiliary Particle Filter dos sistemas lineal y gaussianos, con y sin re-muestreo

- bf_lineal_example: aplicacion del Bootstrap Particle Filter dos sistemas lineal y gaussianos, con y sin re-muestreo

- pendulo: aplicacion del Bootstrap Particle Filter y del Auxiliary Particle Filter para el seguimiento de la posición y la velocidad angular de un péndulo libre


### REQUIREMENTS
Este proyecto requiere la instalación de numpy, scipy y matplotlib. 
