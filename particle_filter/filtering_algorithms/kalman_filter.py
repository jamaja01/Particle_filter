import numpy as np


class KalmanFilter:
    """
    Esta clase ejecuta el filtro de Kalman

    Params:
        -A: matriz transicion
        -H: matriz observacion
        -Q: matriz covarianza proceso
        -R: matriz covarianza medición
        -x0: estado inicial
        -P0: covarianza inicial
        -level: valor para la construcción de los intervalos de confianza   

    Return:
        Devuelve un diccionario con 
        -"estimates": array con los estados estimados para cada tiempo
        -"intervals": array valores para los intervalos de confianza de los estados estimados para cada tiempo
    """
    
    def __init__(self, A, H, Q, R, x0, P0, level = 1.96):
        
        self.A = A 
        self.H = H 
        self.Q = Q 
        self.R = R 

        self.x = x0 
        self.P = P0  

        self.level = level

        self.state_dim = A.shape[0] #Dimension del estado
    

    def reset(self, x0, P0):
        self.x = np.asarray(x0).copy()
        self.P = np.asarray(P0).copy()


    def predict(self):

        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T+ self.Q

        #return self.x, self.P
    
    def update(self, observation):

        # Innovacion
        y = observation - self.H @ self.x

        # Covarianza de la innovacion
        S = self.H @ self.P @ self.H.T + self.R

        # Ganancia de kalman #
        # K = P * H^T * S^-1 => K^T = S^-1 * (P *H^T)^T => S * K^T = (P*H^T)^T 
        PHt = self.P @ self.H.T
        K = np.linalg.solve(S, PHt.T).T #estabilidad

        # Actualizacion del estado
        self.x = self.x + K @ y

        # Actualizacion covarianza del estado 
        # (forma de Joseph para mantener simetria y semidefinitud positiva)
        I = np.eye(self.state_dim)
        J = I - K @ self.H
        self.P = J @ self.P @ J.T + K @ self.R @ K.T

        return self.x, self.P
    

    def interval_confidence(self):

        sigma = np.sqrt(np.diag(self.P))

        return np.column_stack((
            self.x - self.level * sigma,
            self.x + self.level * sigma
        ))


    def step(self, observation):

        self.predict()
        estimate, _ = self.update(observation)

        confidence_intervals = self.interval_confidence()

        return estimate, confidence_intervals


    def run_filter(self, observations):
        
        T = len(observations)

        estimates = np.empty((T, self.state_dim))
        intervals = np.empty((T, self.state_dim, 2))

        for k,y in enumerate(observations):
            
            estimates[k], intervals[k] = self.step(y)

        return {
            "estimates": estimates,
            "intervals": intervals
            }





