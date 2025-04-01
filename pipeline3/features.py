import cv2
import numpy as np

def detect_features(gray_image):
    """
    Detecta keypoints y descriptores con SIFT.
    """
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return keypoints, descriptors


def anms(keypoints, descriptors, N=200, window_size=10):
    """
    Supresión No Máxima Adaptativa (ANMS) según el pseudocódigo del enunciado.
    - Primero detecta los máximos locales en una vecindad.
    - Luego calcula Ri: la mínima distancia al punto más cercano con mayor respuesta.
    - Finalmente selecciona los N puntos con mayor Ri (mejor distribuidos).
    """
    # Paso 1: convertir keypoints a array con (x, y, respuesta)
    kp_array = np.array([kp.pt + (kp.response,) for kp in keypoints])

    maxima_indices = []
    for i, (xi, yi, ri) in enumerate(kp_array):
        is_local_max = True
        for j, (xj, yj, rj) in enumerate(kp_array):
            if i != j:
                if abs(xi - xj) <= window_size and abs(yi - yj) <= window_size:
                    if rj > ri:
                        is_local_max = False
                        break
        if is_local_max:
            maxima_indices.append(i)

    # Paso 2: calcular Ri para cada máximo local
    maxima_kps = [keypoints[i] for i in maxima_indices]
    maxima_descs = [descriptors[i] for i in maxima_indices]
    maxima_array = np.array([kp.pt + (kp.response,) for kp in maxima_kps])

    R = np.full(len(maxima_kps), np.inf)

    for i, (xi, yi, ri) in enumerate(maxima_array):
        for j, (xj, yj, rj) in enumerate(maxima_array):
            if i != j and rj > ri:
                dist = (xj - xi)**2 + (yj - yi)**2
                if dist < R[i]:
                    R[i] = dist

    # Paso 3: quedarnos con los N keypoints con mayor R
    sorted_idx = np.argsort(-R)
    selected_idx = sorted_idx[:N]

    selected_kps = [maxima_kps[i] for i in selected_idx]
    selected_descs = np.array([maxima_descs[i] for i in selected_idx])

    return selected_kps, selected_descs

