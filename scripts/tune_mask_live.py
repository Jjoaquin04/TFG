import cv2
import numpy as np
import sys

def nothing(x):
    pass

def main():
    img_path = 'data/inputs/images/hexagon.png'
    img = cv2.imread(img_path)
    if img is None:
        print("No se pudo cargar la imagen:", img_path)
        sys.exit(1)
        
    #Inicializamos con valores base (actualmente son los valores para la mask 1 en el dict de config)
    h_min, s_min, v_min = 111, 53, 59
    h_max, s_max, v_max = 141, 153, 159

    #Crear ventana con controles deslizantes
    cv2.namedWindow('Tuning Mask', cv2.WINDOW_NORMAL)
    cv2.createTrackbar('H_min', 'Tuning Mask', h_min, 179, nothing)
    cv2.createTrackbar('S_min', 'Tuning Mask', s_min, 255, nothing)
    cv2.createTrackbar('V_min', 'Tuning Mask', v_min, 255, nothing)
    
    cv2.createTrackbar('H_max', 'Tuning Mask', h_max, 179, nothing)
    cv2.createTrackbar('S_max', 'Tuning Mask', s_max, 255, nothing)
    cv2.createTrackbar('V_max', 'Tuning Mask', v_max, 255, nothing)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    print("="*50)
    print(" INTERFAZ DE AJUSTE EN TIEMPO REAL")
    print("="*50)
    print("- Ajusta los valores arrastrando las barras deslizantes.")
    print("- A la izquierda verás la máscara en blanco y negro.")
    print("- A la derecha verás cómo recorta la imagen original.")
    print("- Pulsa 'q' o 'ESC' sobre la ventana de la imagen para terminar y obtener los valores.")
    print("="*50)

    while True:
        h_min = cv2.getTrackbarPos('H_min', 'Tuning Mask')
        s_min = cv2.getTrackbarPos('S_min', 'Tuning Mask')
        v_min = cv2.getTrackbarPos('V_min', 'Tuning Mask')
        
        h_max = cv2.getTrackbarPos('H_max', 'Tuning Mask')
        s_max = cv2.getTrackbarPos('S_max', 'Tuning Mask')
        v_max = cv2.getTrackbarPos('V_max', 'Tuning Mask')
        
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        
        mask = cv2.inRange(hsv, lower, upper)
        
        #Mascara aplicada a la imagen original
        res = cv2.bitwise_and(img, img, mask=mask)
        
        #Juntamos las imagenes para hacer ver comparativa
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        stacked = np.hstack((mask_bgr, res))
        
        cv2.namedWindow('Resultados (Mascara | Imagen Original)', cv2.WINDOW_NORMAL)
        cv2.imshow('Resultados (Mascara | Imagen Original)', stacked)
        
        k = cv2.waitKey(10) & 0xFF
        if k == 27 or k == ord('q'):
            print("\n*** VALORES FINALES PARA COPIAR EN CONFIG.PY ***")
            print(f'"mask1" : [[{h_min}, {s_min}, {v_min}], [{h_max}, {s_max}, {v_max}]]')
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
