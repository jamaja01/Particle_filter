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
├──demos                    # Carpeta con notebooks con demostraciones
|
├──particle_filter
|        |
|        |
|        ├──filtering_algorithms 
|        |          |
|        |          ├──kalman_filter.py #Script para implementar el filtro de Kalman
|        |          |
|        |          ├──particle_filter.py #Script con la clase Particle Filter y las subclases Bootstrap PF y Auxiliary PF
|        |          |  
|        |          ├──resampling_methods.py #Script con las funciones para el remuestreo
|        |        
|        ├──models 
|        |    |
|        |    ├──noise.py #Script con la definición de la clase Noise y sus subclases (GaussianNoise, LaplaceNoise)
|        |    |
|        |    ├──observation.py #Script con la clase ObservationModel y sus subclases aditivas y no aditivas 
|        |    |     
|        |    ├──transition.py #Script con la clase TransitionModel y sus subclases aditivas y no aditivas
|        | 
|        |
|        ├──plots
|        |    |
|        |    ├──plot_comparisons.py #Script para comparar resultados de distintos algoritmos y según su número de partículas
|        |    |     
|        |    ├──plot_results.py #Script para representar los resultados de uno o más algoritmos de filtrado
|        |    |
|        |    ├──plot_weights.py #Script para representar la distribución de pesos
|        |    |
|        |    ├──plot_ESS.py #Script para representar el número efectivo de partículas a través del tiempo  
|        |    |
|        |    ├──plot_particles_diversity.py #Script para representar la diversidad de partículas a través del tiempo
|        |    
|        ├──performance_metrics.py #Script que calcula métricas de rendimiento de algortimos
|        |
|        ├──simulations.py #Script para realizar simulaciones según el modelo especificado
|        |
|        ├──utilities.py #Script con funciones para el procesamiento de los arrays que definen el modelo

```

El nucleo de este proyecto está en los algoritmos de filtrado

- Carpeta [particle_filter]:

    - Carpeta [filtering_algorithms]:

        - Script[particle_filte.py]:
            - Clase [ParticleFilter]: la clase base del proyecto para realizar un filtro de partículas

            - Clase [BootstrapParticleFilter]: subclase de ParticleFilter. Toma como distribución de importancia la distribución a priori $p(\phi_{t} \mid \phi_{t-1})$

            - Clase [AuxiliaryParticleFilter]: subclase de ParticleFilter. Utiliza la observación $y_t$ para identificar cuáles de las partículas en $t-1$ tienen más probabilidades de generar estados compatibles con la observación. En este caso en concreto, se toma como proposal la distribución a priori.

        - Script[kalman_filter.py]: 
            - Clase [KalmanFilter]: clase para implementar el filtro de Kalman     



La carpeta [demos] se tienen los siguientes ejemplos:

- linear_examples (por terminar): aplicacion y comparación de la implementación de los filtros de Kalman, Bootstrap PF y Auxiliary PF para sistemas lineales

- nonadditive (por terminar): aplicacion de ruidos no gaussianos

- Pendulum_Kalman_vs_PF: comparación del filtro de Kalman y el filtro de partículas para el péndulo.

- NonLinearGrowthModel: comparación entre el BPF y el APF según el ruido del sistema y el número de partículas.

- NonLinearGrowthModel_resampling: comparación entre el BPF y el APF según el criterio a la hora de realizar el filtrado.

- Pendulum_smoothness: como afecta el efecto de la discretización al filtro de partículas.

- Bimodal_Model: comparación de los tipos de remuestreo (Multinomial, Systematic y Residual) para un sistema bimodal.


### REQUIREMENTS
Este proyecto requiere la instalación de numpy, scipy y matplotlib. 

Para instalarlos, se recomienda en primer lugar crear un entorno virtual (venv) e instalarlos mediante el comando. 
```
python -m pip install <paquete>
```
Este comando garantiza que se está usando el módulo pip asociado al intérprete de Python.

También se puede instalar a partir del archivo requirements.txt

El proyecto utiliza el archivo pyproject.toml para definir su configuración y dependencias, así como para permitir la importación de los móudulo del proyecto entre sí.