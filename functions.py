from statsmodels.tsa.stattools import coint
import numpy as np

def rolling_coint(series1, series2, window):
    coint_values = np.full(len(series1), np.nan)  
    for i in range(window, len(series1)):
        s1_window = series1[i-window:i]
        s2_window = series2[i-window:i]
        coint_t, p_value, _ = coint(s1_window, s2_window)
        coint_values[i] = p_value  
    return coint_values


