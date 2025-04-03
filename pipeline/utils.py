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
    Dibuja líneas entre puntos coincidentes usando objetos cv2.DMatch.
    - img1, img2: imágenes originales (BGR o RGB o grayscale)
    - kp1, kp2: listas de keypoints de OpenCV
    - matches: lista de tuplas (idx1, idx2)

    Retorna una imagen RGB combinada con los matches dibujados.
    """
    # Asegurar que sean imágenes en color
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if len(img2.shape) == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Crear canvas para mostrar ambas imágenes
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:] = img2

    # Dibujar matches
    for m in matches:
        pt1 = tuple(np.round(kp1[m.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(kp2[m.trainIdx].pt).astype(int) + np.array([w1, 0]))

        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(canvas, pt1, pt2, color, 1)
        cv2.circle(canvas, pt1, 4, color, -1)
        cv2.circle(canvas, pt2, 4, color, -1)

    return canvas

def create_combined_for_plotting(warped0, mask_warped0, canvas1, mask_canvas1, warped2, mask_warped2):
    """ Combina las imágenes y máscaras para visualizar el mix de 'opacidades'"""
    masked0 = (warped0.astype(np.float32) * np.dstack([mask_warped0]*3)).astype(np.uint8)
    masked1 = (canvas1.astype(np.float32) * np.dstack([mask_canvas1]*3)).astype(np.uint8)
    masked2 = (warped2.astype(np.float32) * np.dstack([mask_warped2]*3)).astype(np.uint8)
    combined = cv2.addWeighted(cv2.addWeighted(masked0, 0.7, masked1, 0.7, 0), 0.7, masked2, 0.7, 0)
    return combined