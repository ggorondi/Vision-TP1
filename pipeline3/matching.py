import cv2
import numpy as np
import matplotlib.pyplot as plt
from i308_utils import show_images, imshow

def match_descriptors(desc1, desc2, method="cross_check", ratio_thresh=0.75):
    """
    Empareja descriptores entre dos imágenes con distintos métodos:
    - 'cross_check': validación cruzada
    - 'lowe': ratio test de David Lowe
    - 'both': ambos métodos
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
    
    elif method == "cross_check_and_lowe":
        bf = cv2.BFMatcher(cv2.NORM_L2)
        matchesAtoB = bf.knnMatch(desc1, desc2, k=2)
        goodAtoB = []
        for m, n in matchesAtoB:
            if m.distance < ratio_thresh * n.distance:
                goodAtoB.append(m)
        matchesBtoA = bf.knnMatch(desc2, desc1, k=2)
        goodBtoA = {}
        for m, n in matchesBtoA:
            if m.distance < ratio_thresh * n.distance:
                goodBtoA[m.queryIdx] = m.trainIdx
        final = []
        for m in goodAtoB:
            if goodBtoA.get(m.trainIdx, -1) == m.queryIdx:
                final.append(m)
        return final
    else:
        raise ValueError("Método no válido.")
    


def draw_matches(img1, kp1, img2, kp2, matches, title="Matches"):
    matched_img = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    imshow(matched_img, title=f"{title} - {len(matches)} matches")


