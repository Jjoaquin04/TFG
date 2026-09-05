import json
import os
import math

raw_yolo_path = "data/outputs/json/raw_json/VideoPartido5.json"
interp_yolo_path = "data/outputs/json/interpolated_json/VideoPartido5.json"
gt_path = "data/outputs/json/raw_json/VideoPartido5_annotations.json"

def get_jaccard_and_precision(bb_gt, bb_pred):
    width = max(0, min(bb_gt[2],bb_pred[2]) - max(bb_gt[0], bb_pred[0]))
    height = max(0, min(bb_gt[3], bb_pred[3]) - max(bb_gt[1], bb_pred[1]))

    intersection_area = width*height
    gt_area = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    pred_area = (bb_pred[2] - bb_pred[0]) * (bb_pred[3] - bb_pred[1])
    
    # Índice de Jaccard (IoU = Intersección / Unión)
    jaccard = intersection_area / float(gt_area + pred_area - intersection_area)
    # Precisión Espacial de la caja (Intersección / Área Predicha)
    precision_espacial = intersection_area / float(pred_area) if pred_area > 0 else 0.0
    
    return jaccard, precision_espacial

def evaluate_players(gt_data, yolo_data):
    yolo_players = yolo_data.get('players', {})
    
    total_iou = 0.0
    total_precision_espacial = 0.0
    matches_count = 0
    gt_total_boxes = 0
    pred_total_boxes = 0

    print("\nEVALUACIÓN DE JUGADORES (Jaccard e Precision)\n")

    frames_keys = sorted(gt_data['frames'].keys(), key=int)
    
    for f_index in frames_keys:
        gt_frame_data = gt_data['frames'][f_index]
        yolo_frame_index = str(int(f_index) + 1) # El index de yolo va uno por delante
        
        gt_bboxes = []
        if gt_frame_data.get('players'):
            for p in gt_frame_data['players']:
                if p is not None:
                    gt_bboxes.append([p[0], p[1], p[0] + p[2], p[1] + p[3]])
        
        pred_bboxes = []
        for pid, frames_dict in yolo_players.items():
            if yolo_frame_index in frames_dict:
                p_data = frames_dict[yolo_frame_index]
                pred_bboxes.append([p_data['x_min'], p_data['y_min'], p_data['x_max'], p_data['y_max']])
                
        gt_total_boxes += len(gt_bboxes)
        pred_total_boxes += len(pred_bboxes)
        
        if not gt_bboxes or not pred_bboxes:
            continue
            
        matches_candidates = []
        for i, gt in enumerate(gt_bboxes):
            for j, pred in enumerate(pred_bboxes):
                iou, precision_espacial = get_jaccard_and_precision(gt, pred)
                matches_candidates.append((iou, precision_espacial, i, j))
                
        matches_candidates.sort(key=lambda x: x[0], reverse=True)
        
        matched_gt = set()
        matched_pred = set()
        
        for iou, precision_espacial, i, j in matches_candidates:
            if i not in matched_gt and j not in matched_pred:
                if iou > 0:
                    matched_gt.add(i)
                    matched_pred.add(j)
                    total_iou += iou
                    total_precision_espacial += precision_espacial
                    matches_count += 1

    print(f"Total cajas Ground Truth (anotadas): {gt_total_boxes}")
    print(f"Total cajas YOLO (predicciones): {pred_total_boxes}")
    print(f"Emparejamientos encontrados (Matches): {matches_count}")
    
    if matches_count > 0:
        mean_iou_matches = total_iou / matches_count
        mean_precision_espacial = total_precision_espacial / matches_count
        print(f"\n[IoU] Índice de Jaccard Promedio (Intersección / Unión): {mean_iou_matches:.4f} ({mean_iou_matches*100:.2f}%)")
        print(f"[Precisión Espacial] (Intersección / Área de YOLO): {mean_precision_espacial:.4f} ({mean_precision_espacial*100:.2f}%)")
    else:
        print("\nNo se encontraron emparejamientos.")


def get_iou_ball(bb1, bb2):
    width = max(0, min(bb1[2],bb2[2]) - max(bb1[0], bb2[0]))
    height = max(0, min(bb1[3], bb2[3]) - max(bb1[1], bb2[1]))

    intersection_area = width*height
    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    union = bb1_area + bb2_area - intersection_area
    return intersection_area / float(union) if union > 0 else 0.0

def evaluate_ball_dict(gt_dict, pred_dict, iou_threshold=0.01):
    TP = 0
    FP = 0
    FN = 0
    all_frames = sorted(list(set(list(gt_dict.keys()) + list(pred_dict.keys()))), key=int)

    for f_str in all_frames:
        gt_box = gt_dict.get(f_str)
        pred_boxes = pred_dict.get(f_str, [])

        if gt_box is not None:
            if not pred_boxes:
                FN += 1
            else:
                best_iou = 0
                for p_box in pred_boxes:
                    iou = get_iou_ball(gt_box, p_box)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= iou_threshold:
                    TP += 1
                    FP += (len(pred_boxes) - 1)
                else:
                    FN += 1
                    FP += len(pred_boxes)
        else:
            FP += len(pred_boxes)
            
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return TP, FP, FN, precision, recall, f1

def evaluate_ball(gt_data, raw_yolo_data, interp_yolo_data):
    gt_dict = {}
    for f_str, frame_data in gt_data['frames'].items():
        yolo_f_str = str(int(f_str) + 1)
        ball = frame_data.get('ball')
        if ball is not None:
            gt_dict[yolo_f_str] = [ball[0], ball[1], ball[0]+ball[2], ball[1]+ball[3]]
        else:
            gt_dict[yolo_f_str] = None

    raw_dict = {}
    for item in raw_yolo_data.get('ball', []):
        f_str = str(item['frame'])
        dets = item.get('detections', [])
        boxes = []
        for d in dets:
            boxes.append([d['x_min'], d['y_min'], d['x_max'], d['y_max']])
        raw_dict[f_str] = boxes

    interp_dict = {}
    for item in interp_yolo_data.get('ball', []):
        f_str = str(item['frame'])
        if not math.isnan(item.get('x_min', float('nan'))):
            boxes = [[item['x_min'], item['y_min'], item['x_max'], item['y_max']]]
            interp_dict[f_str] = boxes
        else:
            interp_dict[f_str] = []

    print("\nEVALUACION DE LA PELOTA\n")
    
    print("\n1. RESULTADOS YOLO CRUDO (RAW)")
    print("-" * 30)
    tp, fp, fn, p, r, f1 = evaluate_ball_dict(gt_dict, raw_dict, iou_threshold=0.05)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precisión:    {p:.4f} ({p*100:.2f}%)")
    print(f"Sensibilidad: {r:.4f} ({r*100:.2f}%)")
    print(f"F1-Score:     {f1:.4f} ({f1*100:.2f}%)")

    print("\n2. RESULTADOS YOLO INTERPOLADO")
    print("-" * 30)
    tp, fp, fn, p, r, f1 = evaluate_ball_dict(gt_dict, interp_dict, iou_threshold=0.001)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precisión:    {p:.4f} ({p*100:.2f}%)")
    print(f"Sensibilidad: {r:.4f} ({r*100:.2f}%)")
    print(f"F1-Score:     {f1:.4f} ({f1*100:.2f}%)")

def evaluate_events(gt_data, interp_yolo_data, tolerance=15):
    gt_events = []
    for f_str, v in gt_data['frames'].items():
        if v.get('hit') is not None:
            gt_frame = int(f_str) + 1
            gt_type = v['hit'].get('type', 'unknown')
            gt_events.append({'frame': gt_frame, 'type': gt_type, 'matched': False})

    pred_events = []
    for f_str, ev_list in interp_yolo_data.get('events', {}).items():
        for ev in ev_list:
            p_frame = ev.get('impact_frame', int(f_str))
            p_type = ev.get('type_of_shot', 'unknown')
            pred_events.append({'frame': p_frame, 'type': p_type, 'matched': False})

    gt_events.sort(key=lambda x: x['frame'])
    pred_events.sort(key=lambda x: x['frame'])

    print(f"\nDETECCIÓN DE EVENTOS (Tolerancia: ±{tolerance} frames)")
    
    TP_events = 0
    matched_pairs = []

    for p_idx, p_ev in enumerate(pred_events):
        best_dist = tolerance + 1
        best_gt_idx = -1
        for g_idx, g_ev in enumerate(gt_events):
            if not g_ev['matched']:
                dist = abs(p_ev['frame'] - g_ev['frame'])
                if dist <= tolerance and dist < best_dist:
                    best_dist = dist
                    best_gt_idx = g_idx
                    
        if best_gt_idx != -1:
            gt_events[best_gt_idx]['matched'] = True
            p_ev['matched'] = True
            TP_events += 1
            matched_pairs.append((gt_events[best_gt_idx]['type'], p_ev['type']))

    FP_events = len(pred_events) - TP_events
    FN_events = len(gt_events) - TP_events

    precision_ev = TP_events / (TP_events + FP_events) if (TP_events + FP_events) > 0 else 0.0
    recall_ev = TP_events / (TP_events + FN_events) if (TP_events + FN_events) > 0 else 0.0
    f1_ev = 2 * precision_ev * recall_ev / (precision_ev + recall_ev) if (precision_ev + recall_ev) > 0 else 0.0

    print(f"Total Eventos Reales (GT): {len(gt_events)}")
    print(f"Total Eventos Predichos:   {len(pred_events)}")
    print(f"Emparejamientos (TP):      {TP_events}")
    print(f"Falsos Positivos (FP):     {FP_events}")
    print(f"Falsos Negativos (FN):     {FN_events}")
    print(f"-> Precisión:    {precision_ev:.4f} ({precision_ev*100:.2f}%)")
    print(f"-> Sensibilidad: {recall_ev:.4f} ({recall_ev*100:.2f}%)")
    print(f"-> F1-Score:     {f1_ev:.4f} ({f1_ev*100:.2f}%)")

    print("\n--- CLASIFICACIÓN DEL TIPO DE GOLPE (Solo en los TP detectados) ---")
    if not matched_pairs:
        print("No hay eventos emparejados para evaluar la clasificación.")
        return

    classes = sorted(list(set([str(g) for g, p in matched_pairs] + [str(p) for g, p in matched_pairs])))    
    class_metrics = {}
    for c in classes:
        TP_c = sum(1 for g, p in matched_pairs if str(p) == c and str(g) == c)
        FP_c = sum(1 for g, p in matched_pairs if str(p) == c and str(g) != c)
        FN_c = sum(1 for g, p in matched_pairs if str(g) == c and str(p) != c)
        
        p_c = TP_c / (TP_c + FP_c) if (TP_c + FP_c) > 0 else 0.0
        r_c = TP_c / (TP_c + FN_c) if (TP_c + FN_c) > 0 else 0.0
        f1_c = 2 * p_c * r_c / (p_c + r_c) if (p_c + r_c) > 0 else 0.0
        support = TP_c + FN_c
        
        class_metrics[c] = {'P': p_c, 'R': r_c, 'F1': f1_c, 'Support': support}

    macro_p = sum(m['P'] for m in class_metrics.values()) / len(classes) if classes else 0.0
    macro_f1 = sum(m['F1'] for m in class_metrics.values()) / len(classes) if classes else 0.0
    
    correct_class = sum(1 for g, p in matched_pairs if str(g) == str(p))
    accuracy = correct_class / len(matched_pairs)

    print(f"{'Clase':<15} | {'Precisión':<10} | {'F1-Score':<10} | {'Muestras Reales (Support)'}")
    print("-" * 65)
    for c in classes:
        m = class_metrics[c]
        print(f"{c:<15} | {m['P']:.4f}     | {m['F1']:.4f}     | {m['Support']}")
    print("-" * 65)
    
    print(f"\n[GLOBAL CLASIFICACIÓN]")
    print(f"Precisión (Macro): {macro_p:.4f} ({macro_p*100:.2f}%)")
    print(f"F1-Score (Macro):  {macro_f1:.4f} ({macro_f1*100:.2f}%)")
    print(f"Exactitud (Acc):   {accuracy:.4f} ({accuracy*100:.2f}%)")

def main():
    print("Cargando JSONs...")
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    with open(raw_yolo_path, 'r') as f:
        raw_yolo_data = json.load(f)
    with open(interp_yolo_path, 'r') as f:
        interp_yolo_data = json.load(f)
        
    evaluate_players(gt_data, raw_yolo_data)
    evaluate_ball(gt_data, raw_yolo_data, interp_yolo_data)
    evaluate_events(gt_data, interp_yolo_data, tolerance=15)

if __name__ == "__main__":
    main()
