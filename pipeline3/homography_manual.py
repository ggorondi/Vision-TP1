import numpy as np
from i308_utils import show_images
import cv2

def compute_homography_dlt(pts_src, pts_dst):
    """
    Calcula la matriz de homografía H usando el algoritmo DLT.
    
    Parámetros:
    - pts_src: array de 4 puntos (x, y) en la imagen origen
    - pts_dst: array de 4 puntos (x', y') en la imagen destino
    
    Devuelve:
    - H: matriz de homografía 3x3
    """
    assert pts_src.shape == (4, 2) and pts_dst.shape == (4, 2), "Se necesitan exactamente 4 puntos de cada imagen"

    A = []

    for i in range(4):
        x, y = pts_src[i]
        x_p, y_p = pts_dst[i]

        A.append([-x, -y, -1,  0,  0,  0, x * x_p, y * x_p, x_p])
        A.append([ 0,  0,  0, -x, -y, -1, x * y_p, y * y_p, y_p])

    A = np.array(A)

    # Resolver Ah = 0 usando SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1, :]  # último vector (menor autovalor)
    H = h.reshape((3, 3))

    # Normalizar para que H[2, 2] = 1
    H = H / H[2, 2]

    return H

def show_manual_matches_and_compute_H(matches, resize_factors, images, title_prefix="Manual"):
    """
    Muestra los puntos de correspondencia seleccionados manualmente
    y calcula la homografía entre dos imágenes usando DLT.
    
    Parámetros:
    - matches: diccionario con las correspondencias seleccionadas manualmente
    - resize_factors: factores de redimensionamiento para cada imagen
    
    Devuelve:
    - H: matriz de homografía entre las imágenes seleccionadas
    """
    keys = list(matches.keys())
    idx0 = int(keys[0][-1])  # e.g., 'image0' -> 0
    idx1 = int(keys[1][-1])

    pts0 = np.array(matches[keys[0]]) * resize_factors[idx0]
    pts1 = np.array(matches[keys[1]]) * resize_factors[idx1]

    img0_marked = images[idx0].copy()
    img1_marked = images[idx1].copy()

    for pt in pts0:
        cv2.circle(img0_marked, tuple(pt.astype(int)), 20, (255, 0, 0), -1)
    for pt in pts1:
        cv2.circle(img1_marked, tuple(pt.astype(int)), 20, (255, 0, 0), -1)

    show_images([img0_marked, img1_marked],
                [f'img{idx0}', f'img{idx1}'],
                title=f'Puntos de correspondencia {title_prefix}')

    H = compute_homography_dlt(pts0, pts1)
    print(f"Homografía manual entre img{idx0} y img{idx1}:")
    print(H)

    return H


