import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_transformed_points(img_target, pts_anchor, pts_target, H, title="Transformación manual"):
    """
    Visualiza los puntos transformados con H desde la imagen ancla a la imagen objetivo.
    - img_target: imagen destino (donde van a caer los puntos)
    - pts_anchor: puntos originales de la imagen ancla (N x 2)
    - pts_target: puntos correspondientes reales en imagen destino (N x 2)
    - H: homografía estimada
    """
    # Transformar puntos de la ancla usando H
    pts_homog = np.hstack((pts_anchor, np.ones((len(pts_anchor), 1))))  # (N x 3)
    transformed = (H @ pts_homog.T).T  # (N x 3)
    transformed /= transformed[:, 2][:, np.newaxis]  # normalizar

    # Mostrar
    plt.figure(figsize=(10, 8))
    plt.imshow(img_target, cmap='gray')
    plt.scatter(pts_target[:, 0], pts_target[:, 1], c='red', label='Puntos reales')
    plt.scatter(transformed[:, 0], transformed[:, 1], c='blue', label='Transformados')
    plt.title(title)
    plt.legend()
    plt.xlim(0, img_target.shape[1])
    plt.ylim(img_target.shape[0], 0)
    plt.show()


def draw_matches_ransac(img1, kp1, img2, kp2, matches):
    """
    Dibuja líneas entre puntos coincidentes de dos imágenes.
    - img1, img2: imágenes originales (BGR o RGB o grayscale)
    - kp1, kp2: listas de keypoints de OpenCV
    - matches: lista de tuplas (idx1, idx2)

    Retorna una imagen RGB combinada con los matches dibujados.
    """
    # Convertir imágenes a color si están en escala de grises
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Combinar imágenes horizontalmente
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    # Dibujar matches
    for idx1, idx2 in matches:
        pt1 = tuple(np.round(kp1[idx1].pt).astype(int))
        pt2 = tuple(np.round(kp2[idx2].pt).astype(int) + np.array([w1, 0]))
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

