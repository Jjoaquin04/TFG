import json
import os

import cv2
import numpy as np
from sympy import pprint
import torch

from core.ball import make_windows
from core.ball.heatmap import generate_gaussian_target


class Dataset():

    def __init__(self, ruta_carpeta_imagenes):
        self.data = json.load(open("data/2022_BCN_FinalF_1_ball.json"))
        
        self.etiquetas = {}
        mapa_imagenes = {img['id'] : img['file_name'] for img in self.data['images']}

        for ann in self.data['annotations']:
            bbx = ann['bbox']
            c_x = bbx[0] + bbx[2]/2.0
            c_y = bbx[1] + bbx[3]/2.0

            image_id = mapa_imagenes[ann['image_id']]

            if image_id is None:
                self.etiquetas[image_id] = [-1,-1]
            else:
                self.etiquetas[image_id] = [c_x, c_y]

        self.listas_imagenes = sorted(img['file_name'] for img in self.data['images'])

    def __len__(self): 

        return len(self.listas_imagenes) - 2

    def __getItem__(self, idx):

        windows_maker = make_windows()

        three_images_names = [
            self.listas_imagenes[idx],
            self.listas_imagenes[idx+1],
            self.listas_imagenes[idx+2]
        ]

        for nombre in three_images_names:
            frame = cv2.imread(f"data/inputs/images/{nombre}")
            tensor = windows_maker.update_frames(frame)
        sandwich_listo = None
        
        # Leemos las 3 fotos del disco duro y las metemos en la cinta transportadora
        for nombre in three_images_names:
            ruta_completa = os.path.join(self.ruta_imagenes, nombre)
            fotograma = cv2.imread(ruta_completa)
            
            # Si OpenCV no encuentra la foto, metemos una pantalla negra por si acaso
            if fotograma is None:
                fotograma = np.zeros((1080, 1920, 3), dtype=np.uint8)
                
            sandwich_listo = windows_maker.update_frames(fotograma)
            
        # ==========================================
        # EL OBJETIVO (El examen para la IA)
        # ==========================================
        # La IA tiene que adivinar dónde está la pelota en la ÚLTIMA foto de las 3 (la foto 7)
        nombre_ultima_foto = three_images_names[2]
        
        # Buscamos en nuestro diccionario si sabemos dónde está la pelota en esa foto
        if nombre_ultima_foto in self.etiquetas:
            centro_x, centro_y = self.etiquetas[nombre_ultima_foto]
        else:
            # Si no está etiquetada, no hay pelota
            centro_x, centro_y = -1.0, -1.0
            
        # IMPORTANTE: Como hemos achicado las fotos a 640x360 para la IA, 
        # tenemos que achicar también las coordenadas originales (que estaban en 1920x1080)
        escala_x = 640 / 1920
        escala_y = 360 / 1080
        centro_x_achicado = centro_x * escala_x if centro_x > 0 else -1.0
        centro_y_achicado = centro_y * escala_y if centro_y > 0 else -1.0
        
        # ¡Pintamos la diana!
        heatmap_objetivo = generate_gaussian_target(
            width=640, height=360, 
            c_x=centro_x_achicado, c_y=centro_y_achicado, sigma=2.5
        )
        
        # Convertimos todo a formato PyTorch (Tensores) para que la IA lo entienda
        tensor_x = torch.from_numpy(sandwich_listo)
        
        # El heatmap tiene que tener un "canal" extra [1, 360, 640]
        tensor_y = torch.from_numpy(heatmap_objetivo).unsqueeze(0)
        
        return tensor_x, tensor_y



    

def main():
    dataset = Dataset()

if __name__ == "__main__":    main()