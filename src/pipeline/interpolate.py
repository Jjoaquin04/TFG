import pandas as pd
import json
import os
import config

def interpolate(raw_json_path: str):
    """
    Lee el fichero JSON crudo ('data/json/extend/...') generado en la fase extract,
    interpola numéricamente los frames perdidos de la pelota (bbx y posición real),
    y guarda el resultado en un nuevo JSON ('data/json/reduced/...').
    """
    raw_path = raw_json_path
    file_name = os.path.basename(raw_path)
    interp_path = os.path.join(config.INTERP_JSON_PATH, "interpolated_json", file_name)
    
    print(f"Loading raw detections from {raw_path}...")
    with open(raw_path, 'r') as f:
        data = json.load(f)
        
    ball_history = data.get('ball', [])
    if not ball_history:
        print("No ball data found.")
        return None

    # Convierte la info de la pelota en un DataFrame indicando el índice temporal
    df_ball = pd.DataFrame(ball_history).set_index('frame')
    
    # Interpola de forma lineal para cubrir todos los frames sin detección (NaN -> Números)
    # limit_direction="both" asegura que cubrirá incluso si falla al puro inicio o final
    df_ball_interp = df_ball.interpolate(method='linear', limit_direction='both')
    
    # Restaura los floats en un formato seguro para JSON de nuevo a lista de diccionarios
    data['ball'] = df_ball_interp.reset_index().to_dict(orient='records')

    # Guarda el nuevo JSON ya con valores en todos los frames
    os.makedirs(os.path.dirname(interp_path), exist_ok=True)
    with open(interp_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Interpolation finished! Saved cleaned data to {interp_path}")
    return interp_path
