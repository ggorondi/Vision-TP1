import cv2
import numpy as np
from i308_utils import show_images, imshow
from pipeline.utils import create_combined_for_plotting

def obtener_esquinas(img):
    """ Devuelve las coordenadas de las esquinas de una imagen. """
    h, w = img.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

def transformar_esquinas(corners, H):
    """ Transforma las esquinas de una imagen usando una homografía H. """
    corners_h = np.hstack([corners, np.ones((corners.shape[0], 1))])
    transformed = (H @ corners_h.T).T
    return transformed[:, :2] / transformed[:, 2][:, np.newaxis]

def tamaño_pano(img0, img1, img2, H01, H12):
    """ Calcula el tamaño del panorama final y la traslación necesaria. """
    corners0 = obtener_esquinas(img0)
    corners1 = obtener_esquinas(img1)
    corners2 = obtener_esquinas(img2)

    t_corners0 = transformar_esquinas(corners0, H01)
    t_corners2 = transformar_esquinas(corners2, H12)

    all_corners = np.vstack([t_corners0, corners1, t_corners2])

    x_min, y_min = np.min(all_corners, axis=0)
    x_max, y_max = np.max(all_corners, axis=0)

    width = int(np.ceil(x_max - x_min))
    height = int(np.ceil(y_max - y_min))

    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])
    return width, height, translation

def ensure_rgb(img):
    """ Convierte una imagen a RGB si no lo es. """
    if len(img.shape) == 2 or img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

# def generate_gradient_mask(warped):
#     """ Genera una máscara de gradiente para una imagen transformada. """
#     mask = np.where(warped > 0, 1, 0).astype(np.uint8)
#     gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
#     dist = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
#     norm = dist / (dist.max() + 1e-6)
#     imshow(norm, "Distancia transformada")
#     return np.repeat(norm[:, :, np.newaxis], 3, axis=2)

def compute_edge_mask(image):
    """ Calcula un mask de 'opacidad' basado en la distancia a los bordes de la imagen. """
    h, w = image.shape[:2]
    x = np.arange(w)
    y = np.arange(h)
    xv, yv = np.meshgrid(x, y)
    dist_left = xv
    dist_right = w - 1 - xv
    dist_top = yv
    dist_bottom = h - 1 - yv
    dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_top, dist_bottom))
    normalized = dist.astype(np.float32) / np.max(dist)
    return normalized

# def stitch_images_with_blending(img0, img1, img2, H01, H12):
#     """ Une tres imágenes usando homografías H01 y H12. """
#     img0 = ensure_rgb(img0)
#     img1 = ensure_rgb(img1)
#     img2 = ensure_rgb(img2)

#     width, height, T = tamaño_pano(img0, img1, img2, H01, H12)

#     panorama = np.zeros((height, width, 3), dtype=np.float32)
#     weights = np.zeros((height, width, 3), dtype=np.float32)

#     H0 = T @ H01
#     H2 = T @ H12

#     warped0 = cv2.warpPerspective(img0, H0, (width, height))
#     warped2 = cv2.warpPerspective(img2, H2, (width, height))

#     mask0 = generate_gradient_mask(warped0)
#     mask2 = generate_gradient_mask(warped2)

#     panorama += warped0 * mask0
#     weights += mask0

#     panorama += warped2 * mask2
#     weights += mask2

#     offset_x = int(T[0, 2])
#     offset_y = int(T[1, 2])

#     h1, w1 = img1.shape[:2]
#     x1, y1 = offset_x, offset_y

#     blend_mask = np.zeros((height, width), dtype=np.uint8)
#     blend_mask[y1:y1 + h1, x1:x1 + w1] = 1
#     dist1 = cv2.distanceTransform(blend_mask, cv2.DIST_L2, 5)
#     dist1_crop = dist1[y1:y1 + h1, x1:x1 + w1]
#     dist1_norm = dist1_crop / (dist1_crop.max() + 1e-6)
#     dist1_rgb = np.repeat(dist1_norm[:, :, np.newaxis], 3, axis=2)

#     panorama[y1:y1 + h1, x1:x1 + w1] += img1 * dist1_rgb
#     weights[y1:y1 + h1, x1:x1 + w1] += dist1_rgb

#     blended = (panorama / np.maximum(weights, 1e-6)).astype(np.uint8)
#     return blended


def stitch_images_with_blending(fot0, fot1, fot2, H_01, H_21):
    """ Aplica homografías y crea la panorámica final de tres fotos. """
    h0, w0 = fot0.shape[:2]
    h1, w1 = fot1.shape[:2]
    h2, w2 = fot2.shape[:2]
    mask0 = compute_edge_mask(fot0)
    mask1 = compute_edge_mask(fot1)
    mask2 = compute_edge_mask(fot2)
    corners0 = np.array([[0,0],[w0,0],[w0,h0],[0,h0]], dtype=np.float32).reshape(-1,1,2)
    warped_corners0 = cv2.perspectiveTransform(corners0, H_01).reshape(-1,2)
    corners1 = np.array([[0,0],[w1,0],[w1,h1],[0,h1]], dtype=np.float32)
    corners2 = np.array([[0,0],[w2,0],[w2,h2],[0,h2]], dtype=np.float32).reshape(-1,1,2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H_21).reshape(-1,2)
    all_corners = np.vstack((warped_corners0, corners1, warped_corners2))
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    tx, ty = -x_min, -y_min
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    T = np.array([[1, 0, tx],[0, 1, ty],[0, 0, 1]])
    warped0 = cv2.warpPerspective(fot0, T @ H_01, (canvas_width, canvas_height))
    warped_mask0 = cv2.warpPerspective(mask0, T @ H_01, (canvas_width, canvas_height))
    warped2 = cv2.warpPerspective(fot2, T @ H_21, (canvas_width, canvas_height))
    warped_mask2 = cv2.warpPerspective(mask2, T @ H_21, (canvas_width, canvas_height))
    canvas1 = np.zeros((canvas_height, canvas_width, 3), dtype=fot1.dtype)
    canvas_mask1 = np.zeros((canvas_height, canvas_width), dtype=mask1.dtype)
    canvas1[ty:ty+h1, tx:tx+w1] = fot1
    canvas_mask1[ty:ty+h1, tx:tx+w1] = mask1
    W0 = np.dstack([warped_mask0]*3)
    W1 = np.dstack([canvas_mask1]*3)
    W2 = np.dstack([warped_mask2]*3)
    
    combined = create_combined_for_plotting(warped0, warped_mask0, canvas1, canvas_mask1, warped2, warped_mask2)

    blended = (warped0.astype(np.float32)*W0 + canvas1.astype(np.float32)*W1 + warped2.astype(np.float32)*W2)/(W0+W1+W2+1e-6)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return warped0, canvas1, warped2, warped_mask0, canvas_mask1, warped_mask2, combined, blended
