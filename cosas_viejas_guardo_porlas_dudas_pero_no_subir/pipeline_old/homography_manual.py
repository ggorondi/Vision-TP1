import numpy as np

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



