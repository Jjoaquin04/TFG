# Padel Tracking & Event Classification

Este proyecto es un Trabajo de Fin de Grado (TFG) orientado al rastreo de jugadores y pelota en vídeos de pádel, así como a la clasificación automática de golpes (saques, remates, globos) a partir de técnicas de Visión por Computador e Inteligencia Artificial.

## Estructura de Ejecución (Pipeline)

El proyecto está diseñado para ejecutarse en tres fases secuenciales independientes. Esto permite procesar la inferencia masiva una sola vez y luego iterar rápidamente sobre la limpieza de datos o el renderizado.

### 1. Fase de Extracción (`--extract`)
Lee el vídeo de entrada, ejecuta las redes neuronales de detección/pose y rastrea todos los objetos (pista, jugadores, pelota). Genera un archivo `.json` en crudo con toda la información extraída.
```bash
python src/main.py --extract data\inputs\videos\tu_video.mp4
```

### 2. Fase de Post-Procesado (`--postprocessing`)
Toma el JSON crudo generado en la fase anterior, suaviza las trayectorias de la pelota, aplica interpolaciones, determina qué jugador golpea la pelota en cada instante y clasifica el tipo de golpe (Service, Smash, Lob).
```bash
python src/main.py --postprocessing data\outputs\json\raw_json\tu_video.json
```

### 3. Fase de Renderizado (`--render`)
Coge el vídeo original y el JSON final procesado, y dibuja las cajas delimitadoras, el minimapa bidimensional (Minicourt) y las trayectorias de los golpes en un nuevo vídeo de salida.
```bash
python src/main.py --render data\inputs\videos\tu_video.mp4 data\outputs\json\tu_video_interpolated.json
```

---

## Modelos de IA y Optimización (OpenVINO)

El proyecto utiliza modelos basados en **Ultralytics YOLOv8** para:
- Detección de la Pista (Keypoints)
- Pose de los Jugadores (Pose)
- Detección de la Pelota (Bounding Box)

### Configuración de Modelos (`src/config.py`)
En el archivo `src/config.py` se definen las rutas de los modelos que se utilizarán en la fase de extracción.

Por defecto, el código utiliza modelos estándar de PyTorch (`.pt`). Sin embargo, si estás ejecutando este código en un procesador **Intel (CPU)** o una gráfica integrada **Intel Iris Xe**, se recomienda encarecidamente exportar los modelos al formato **OpenVINO** para agilizar radicalmente el tiempo de inferencia.

**¿Cómo exportar a OpenVINO?**
Ejecuta los siguientes comandos en tu terminal (requiere instalar `openvino`):
```bash
yolo export model=models/yolov8-court-keypoint.pt format=openvino
yolo export model=models/yolov8-player-pose.pt format=openvino
yolo export model=models/yolov8-ball-bbx.pt format=openvino
```

**¿Cómo usarlos en el proyecto?**
Abre `src/config.py` y comenta las líneas de los modelos `.pt` para descomentar las rutas hacia las nuevas carpetas generadas con terminación `_openvino_model/`.

---

## Librerías Requeridas

Asegúrate de instalar las dependencias básicas para ejecutar el proyecto (recomendado usar entorno virtual):

```bash
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pandas
# Para exportar a OpenVINO (opcional pero recomendado en CPUs Intel):
pip install openvino
```
