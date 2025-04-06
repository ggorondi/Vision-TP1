import cv2
import numpy as np
from i308_utils import show_images, imshow

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

def create_combined_for_plotting(warped0, mask_warped0, canvas1, mask_canvas1, warped2, mask_warped2):
    """ 
    Combina las imágenes y máscaras para visualizar el mix de 'opacidades' en el canvas panorama
    (no arma el panorama final, solo muestra la mezcla de imágenes)
    
    Retorna la imagen combinada.
    """
    masked0 = (warped0.astype(np.float32) * np.dstack([mask_warped0]*3)).astype(np.uint8)
    masked1 = (canvas1.astype(np.float32) * np.dstack([mask_canvas1]*3)).astype(np.uint8)
    masked2 = (warped2.astype(np.float32) * np.dstack([mask_warped2]*3)).astype(np.uint8)
    combined = cv2.addWeighted(cv2.addWeighted(masked0, 0.7, masked1, 0.7, 0), 0.7, masked2, 0.7, 0)
    return combined

def stitch_images_with_blending(fot0, fot1, fot2, H_01, H_21):
    """ 
    Aplica homografías y crea una serie de imágenes para visualizar el proceso, y la panorámica final de tres fotos. 
    
    Devuelve un diccionario con las siguientes imágenes:
    - "warped0": imagen fot0 transformada
    - "warped_mask0": máscara de fot0 transformada
    - "canvas1": imagen fot1 transformada
    - "canvas_mask1": máscara de fot1 transformada
    - "warped2": imagen fot2 transformada
    - "warped_mask2": máscara de fot2 transformada
    - "combined": imagen combinada de fotos maskeadas para visualización
    - "blended": imagen final del panorama
    """
    
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
    
    # determinar tamaño del pano
    all_corners = np.vstack((warped_corners0, corners1, warped_corners2))
    x_min, y_min = np.floor(all_corners.min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0)).astype(int)
    tx, ty = -x_min, -y_min
    canvas_width = x_max - x_min
    canvas_height = y_max - y_min
    
    # matriz de Traslación canvas pano
    T = np.array([[1, 0, tx],[0, 1, ty],[0, 0, 1]])
    
    warped0 = cv2.warpPerspective(fot0, T @ H_01, (canvas_width, canvas_height))
    warped_mask0 = cv2.warpPerspective(mask0, T @ H_01, (canvas_width, canvas_height))
    warped2 = cv2.warpPerspective(fot2, T @ H_21, (canvas_width, canvas_height))
    warped_mask2 = cv2.warpPerspective(mask2, T @ H_21, (canvas_width, canvas_height))
    
    canvas1 = np.zeros((canvas_height, canvas_width, 3), dtype=fot1.dtype)
    canvas_mask1 = np.zeros((canvas_height, canvas_width), dtype=mask1.dtype)
    canvas1[ty:ty+h1, tx:tx+w1] = fot1
    canvas_mask1[ty:ty+h1, tx:tx+w1] = mask1
    
    # masks a 3 canales
    W0 = np.dstack([warped_mask0]*3)
    W1 = np.dstack([canvas_mask1]*3)
    W2 = np.dstack([warped_mask2]*3)
    
    # (para mostrar superposición de imágenes con opacidad, no es el resultado final)
    combined = create_combined_for_plotting(warped0, warped_mask0, canvas1, canvas_mask1, warped2, warped_mask2)

    # panorama final
    blended = (warped0.astype(np.float32)*W0 + canvas1.astype(np.float32)*W1 + warped2.astype(np.float32)*W2)/(W0+W1+W2+1e-6)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return {
        'warped0': warped0,
        'warped_mask0': warped_mask0,
        'canvas1': canvas1,
        'canvas_mask1': canvas_mask1,
        'warped2': warped2,
        'warped_mask2': warped_mask2,
        'combined': combined,
        'blended': blended
    }
