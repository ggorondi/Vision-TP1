import cv2
import os
import numpy as np

def resize(img, w):
    """
    Redimensiona una imagen manteniendo la relación de aspecto,
    usando un ancho deseado `w`.
    """
    w_, h_ = img.shape[1], img.shape[0]
    h = int(h_ * w / w_)
    return cv2.resize(img, (w, h))


def load_images_from_folder(folder_path, target_width=None):
    """
    Carga todas las imágenes .jpg o .png de un folder, las redimensiona a un ancho fijo
    y las convierte a escala de grises.
    """
    images_color = []
    images_gray = []
    resize_factors = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join(folder_path, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            # Resize a ancho deseado
            if target_width is not None:
                # Calcula el factor de redimensionamiento
                h, w = img.shape[:2]
                resize_factor = target_width / w
                resize_factors.append(resize_factor)
                img = resize(img, target_width)
            else:
                resize_factors.append(1.0)

            # Escala de grises
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            images_color.append(img)
            images_gray.append(gray)
            

    return images_color, images_gray, resize_factors

