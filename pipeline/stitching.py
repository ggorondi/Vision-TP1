import cv2
import numpy as np

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

def generate_gradient_mask(warped):
    """ Genera una máscara de gradiente para una imagen transformada. """
    mask = np.where(warped > 0, 1, 0).astype(np.uint8)
    gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    dist = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
    norm = dist / (dist.max() + 1e-6)
    return np.repeat(norm[:, :, np.newaxis], 3, axis=2)

def stitch_images_with_blending(img0, img1, img2, H01, H12):
    """ Une tres imágenes usando homografías H01 y H12. """
    img0 = ensure_rgb(img0)
    img1 = ensure_rgb(img1)
    img2 = ensure_rgb(img2)

    width, height, T = tamaño_pano(img0, img1, img2, H01, H12)

    panorama = np.zeros((height, width, 3), dtype=np.float32)
    weights = np.zeros((height, width, 3), dtype=np.float32)

    H0 = T @ H01
    H2 = T @ H12

    warped0 = cv2.warpPerspective(img0, H0, (width, height))
    warped2 = cv2.warpPerspective(img2, H2, (width, height))

    mask0 = generate_gradient_mask(warped0)
    mask2 = generate_gradient_mask(warped2)

    panorama += warped0 * mask0
    weights += mask0

    panorama += warped2 * mask2
    weights += mask2

    offset_x = int(T[0, 2])
    offset_y = int(T[1, 2])

    h1, w1 = img1.shape[:2]
    x1, y1 = offset_x, offset_y

    blend_mask = np.zeros((height, width), dtype=np.uint8)
    blend_mask[y1:y1 + h1, x1:x1 + w1] = 1
    dist1 = cv2.distanceTransform(blend_mask, cv2.DIST_L2, 5)
    dist1_crop = dist1[y1:y1 + h1, x1:x1 + w1]
    dist1_norm = dist1_crop / (dist1_crop.max() + 1e-6)
    dist1_rgb = np.repeat(dist1_norm[:, :, np.newaxis], 3, axis=2)

    panorama[y1:y1 + h1, x1:x1 + w1] += img1 * dist1_rgb
    weights[y1:y1 + h1, x1:x1 + w1] += dist1_rgb

    blended = (panorama / np.maximum(weights, 1e-6)).astype(np.uint8)
    return blended
