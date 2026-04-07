import numpy as np
import math

"""
Atributes:
    - width: Ancho del heatmap
    - height: Alto del heatmap
    - center_x: Coordenada x del centro de la pelota
    - center_y: Coordenada y del centro de la pelota
    - sigma: Desviación estándar de la gaussiana (controla el tamaño de la mancha)

"""

def generate_gaussian_target(width,height,center_x,center_y, sigma=2.5): 
    
    heatmap = np.zeros((height,width), dtype=np.float32)

    if center_x < 0 or center_y < 0 or math.isnan(center_x) or math.isnan(center_y):
        return heatmap
    
    radius = int(3 * sigma)

    # Coordenadas de la roi (región de interés) alrededor del centro de la pelota
    x0 = max(0, int(center_x) - radius)
    y0 = max(0, int(center_y) - radius)
    x1 = min(width, int(center_x) + radius)
    y1 = min(height, int(center_y) + radius)

    for y in range(y0, y1):
        for x in range(x0, x1):
            distancia_squared = (x - center_x) ** 2 + (y - center_y) ** 2

            probability = math.exp(-distancia_squared / (2 * sigma ** 2))
            heatmap[y, x] = probability

    return heatmap
