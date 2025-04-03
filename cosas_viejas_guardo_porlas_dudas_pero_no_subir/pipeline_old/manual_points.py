import cv2
import matplotlib.pyplot as plt
import numpy as np
from preprocessing import resize


def seleccionar_correspondencias(img_path_1, img_path_2, ancho=640):
    """
    Permite seleccionar manualmente 4 puntos en cada una de dos imágenes,
    y devuelve las coordenadas seleccionadas.
    """
    # Cargar y redimensionar imágenes
    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)

    img1 = resize(img1, ancho)
    img2 = resize(img2, ancho)

    # Convertir a RGB para visualización con matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Mostrar imagen 1 y seleccionar puntos
    print("Seleccioná 4 puntos en la primera imagen (imagen ancla)")
    plt.figure(figsize=(8, 6))
    plt.imshow(img1_rgb)
    pts1 = plt.ginput(4, timeout=0)
    plt.close()

    # Mostrar imagen 2 y seleccionar puntos
    print("Seleccioná 4 puntos en la segunda imagen (imagen objetivo)")
    plt.figure(figsize=(8, 6))
    plt.imshow(img2_rgb)
    pts2 = plt.ginput(4, timeout=0)
    plt.close()

    # Mostrar resultados
    print("Puntos seleccionados en imagen 1:", pts1)
    print("Puntos seleccionados en imagen 2:", pts2)

    return np.array(pts1), np.array(pts2)


# Ejemplo de uso (ejecutar solo si corrés este script directamente)
if __name__ == '__main__':
    # anchor_pts, target_pts = seleccionar_correspondencias("img/udesa_1.jpg", "img/udesa_0.jpg")
    # anchor_pts, target_pts = seleccionar_correspondencias("img/udesa_2.jpg", "img/udesa_0.jpg")
    anchor_pts, target_pts = seleccionar_correspondencias("img/cuadro_1.jpg", "img/cuadro_0.jpg")
    # anchor_pts, target_pts = seleccionar_correspondencias("img/cuadro_2.jpg", "img/cuadro_0.jpg")

