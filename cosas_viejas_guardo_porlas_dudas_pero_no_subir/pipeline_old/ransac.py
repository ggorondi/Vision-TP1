import numpy as np
from pipeline.homography_manual import compute_homography_dlt

def ransac_homography(pts_src, pts_dst, num_iters=1000, threshold=5.0):
    """
    Estima una homografía robusta usando RANSAC.
    
    Parámetros:
    - pts_src, pts_dst: puntos emparejados (N x 2)
    - num_iters: cantidad de iteraciones
    - threshold: distancia máxima para considerar un inlier
    
    Devuelve:
    - mejor_H: homografía con más inliers
    - inlier_mask: array booleano de tamaño N con True para inliers
    """
    assert pts_src.shape == pts_dst.shape
    N = pts_src.shape[0]
    best_inliers = 0
    best_H = None
    best_mask = None

    for _ in range(num_iters):
        # Elegir 4 pares al azar
        idx = np.random.choice(N, 4, replace=False)
        src_sample = pts_src[idx]
        dst_sample = pts_dst[idx]

        # Calcular H con DLT
        H = compute_homography_dlt(src_sample, dst_sample)

        # Transformar todos los puntos
        pts_src_h = np.hstack((pts_src, np.ones((N, 1))))
        projected = (H @ pts_src_h.T).T
        projected /= projected[:, 2][:, np.newaxis]

        # Calcular distancias a los puntos reales
        distances = np.linalg.norm(projected[:, :2] - pts_dst, axis=1)

        # Crear máscara de inliers
        mask = distances < threshold
        num_inliers = np.sum(mask)

        if num_inliers > best_inliers:
            best_inliers = num_inliers
            best_H = H
            best_mask = mask

    return best_H, best_mask
