import cv2
import numpy as np
import argparse

cuadrantes_per_frame = {
    "img1": [ # Imagen morada
        [200, 450, 250, 500],   
        [1400, 450, 1450, 500], 
        [300, 850, 350, 900],  
        [1500, 850, 1550, 900]  
    ],
    "img2": [ # Imagen azul 
        [450, 350, 500, 400],   
        [1650, 350, 1700, 400], 
        [450, 750, 500, 800],   
        [1750, 750, 1800, 800]  
    ],
    "img3": [ # Imagen azul
        [250, 400, 300, 450],   
        [1500, 400, 1550, 450], 
        [250, 800, 300, 850],   
        [1600, 800, 1650, 850]  
    ],
    "img4": [ # Imagen azul 
        [250, 400, 300, 450],   
        [1500, 400, 1550, 450],
        [800, 800, 850, 850],   
        [1000, 800, 1050, 850]
    ]
}

def extract_color_mask(img1_path, img2_path, img3_path, img4_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img3 = cv2.imread(img3_path)
    img4 = cv2.imread(img4_path)
    images_dict = {
        "img1": img1, 
        "img2": img2, 
        "img3": img3, 
        "img4": img4
    }
    #Actualizacion: para la pista morada se ha usado el file scripts/tune_mask_live.py para mejorar su máscara
    h_margin = 15
    s_margin = 50
    v_margin = 50
    for img_name, img in images_dict.items():
        
        if img is None:
            continue

        pixels = []
        windows = cuadrantes_per_frame[img_name]
        img_debug = img.copy()
        img_hsv = cv2.cvtColor(img_debug, cv2.COLOR_BGR2HSV)

        for (x1, y1, x2, y2) in windows:
            frame_window = img_hsv[y1:y2, x1:x2]
            pixles_window = np.reshape(frame_window, (-1, 3))
            pixels.append(pixles_window)
    
        #Trnasformamos las listas para cada window en una sola lista
        all_pixels = np.vstack(pixels)
        
        median = np.median(all_pixels, axis=0) #axis=0 para que lo calcule columna por columna

        median_hue = median[0]
        median_saturation = median[1]
        median_brightness = median[2]

        lower_hue = max(0, median_hue - h_margin)
        lower_saturation = max(0, median_saturation - s_margin)
        lower_brightness = max(0, median_brightness - v_margin)

        upper_hue = min(179, median_hue + h_margin)
        upper_saturation = min(255, median_saturation + s_margin)
        upper_brightness = min(255, median_brightness + v_margin)

        lower = np.array([lower_hue, lower_saturation, lower_brightness],dtype = np.uint8)
        upper = np.array([upper_hue, upper_saturation, upper_brightness], dtype = np.uint8)

        print(f"{lower.tolist()}")
        print(f"{upper.tolist()}")
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path1', type=str, help='Path de img1')
    parser.add_argument('path2', type=str, help='Path de img2')
    parser.add_argument('path3', type=str, help='Path de img3')
    parser.add_argument('path4', type=str, help='Path de img4')
    args = parser.parse_args()
    extract_color_mask(args.path1, args.path2, args.path3, args.path4)


