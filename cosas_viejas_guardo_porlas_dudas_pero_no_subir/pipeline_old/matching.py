import cv2
import numpy as np
import matplotlib.pyplot as plt

def match_descriptors(desc1, desc2, method="cross_check", ratio_thresh=0.75):
    """
    Empareja descriptores entre dos imágenes con distintos métodos:
    - 'cross_check': validación cruzada
    - 'lowe': ratio test de David Lowe
    """
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=(method == "cross_check"))

    if method == "cross_check":
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches

    elif method == "lowe":
        matches = bf.knnMatch(desc1, desc2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches

    else:
        raise ValueError("Método no válido. Elegí entre 'cross_check' o 'lowe'")
    


def draw_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(20, 10))
    plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


