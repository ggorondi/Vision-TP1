import cv2
import numpy as np

def anms(keypoints, descriptors, N=1000):
    """Aplica Adaptive Non-Maximal Suppression (ANMS) para seleccionar los N keypoints más relevantes."""
    indexed_kps = sorted(enumerate(keypoints), key=lambda x: x[1].response, reverse=True)
    radii = [float('inf')] * len(indexed_kps)
    for i in range(len(indexed_kps)):
        if i == 0:
            radii[i] = float('inf')
        else:
            (x_i, y_i) = indexed_kps[i][1].pt
            min_dist = float('inf')
            for j in range(i):
                (x_j, y_j) = indexed_kps[j][1].pt
                dist = (x_i - x_j)**2 + (y_i - y_j)**2
                if dist < min_dist:
                    min_dist = dist
            radii[i] = np.sqrt(min_dist)
    kps_radii = sorted([(idx, kp, r) for (idx, kp), r in zip(indexed_kps, radii)], key=lambda x: x[2], reverse=True)
    selected = kps_radii[:N]
    selected_kps = [item[1] for item in selected]
    selected_indices = [item[0] for item in selected]
    selected_desc = descriptors[selected_indices, :]
    return selected_kps, selected_desc

def cross_check_and_lowe_match(descA, descB, lowe_ratioA=0.8, lowe_ratioB=0.8):
    """ Matchea descriptores con Brute Force y filtra matches usando Lowe's ratio test y cross-checking. """
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matchesAtoB = bf.knnMatch(descA, descB, k=2)
    goodAtoB = []
    for m, n in matchesAtoB:
        if m.distance < lowe_ratioA * n.distance:
            goodAtoB.append(m)
    matchesBtoA = bf.knnMatch(descB, descA, k=2)
    goodBtoA = {}
    for m, n in matchesBtoA:
        if m.distance < lowe_ratioB * n.distance:
            goodBtoA[m.queryIdx] = m.trainIdx
    final = []
    for m in goodAtoB:
        if goodBtoA.get(m.trainIdx, -1) == m.queryIdx:
            final.append(m)
    return final

def extract_matched_points(kp1, kp2, matches):
    """ separa los matches de tipo cv2.DMatch en dos listas de puntos. """
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    return pts1, pts2

def apply_homography(H, pts):
    """ Aplica la homografía H a los puntos pts. """
    pts_h = np.hstack([pts, np.ones((pts.shape[0], 1))])
    proj = (H @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2, np.newaxis]
    return proj

def compute_homography(pts1, pts2):
    """ Calcula la homografía entre dos conjuntos de puntos. """
    assert pts1.shape[0] >= 4 and pts1.shape == pts2.shape
    A = []
    for i in range(pts1.shape[0]):
        x, y = pts1[i]
        xp, yp = pts2[i]
        A.append([-x, -y, -1,  0,  0, 0, x*xp, y*xp, xp])
        A.append([ 0,  0,  0, -x, -y, -1, x*yp, y*yp, yp])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H / H[2, 2]

def ransac_match_filter(pts1, pts2, threshold=5.0, max_iter=1000):
    """ Aplica RANSAC para filtrar matches a un conjunto de 'inliers'"""
    best_inlier_mask = None
    max_inliers = 0
    N = pts1.shape[0]
    for _ in range(max_iter):
        idx = np.random.choice(N, 4, replace=False)
        H_candidate = compute_homography(pts1[idx], pts2[idx])
        pts1_proj = apply_homography(H_candidate, pts1)
        dists = np.linalg.norm(pts2 - pts1_proj, axis=1)
        inliers = dists < threshold
        num_inliers = np.sum(inliers)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_inlier_mask = inliers
    return best_inlier_mask

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
    """ Combina las imágenes y máscaras para visualizar el mix de 'opacidades'"""
    masked0 = (warped0.astype(np.float32) * np.dstack([mask_warped0]*3)).astype(np.uint8)
    masked1 = (canvas1.astype(np.float32) * np.dstack([mask_canvas1]*3)).astype(np.uint8)
    masked2 = (warped2.astype(np.float32) * np.dstack([mask_warped2]*3)).astype(np.uint8)
    combined = cv2.addWeighted(cv2.addWeighted(masked0, 0.7, masked1, 0.7, 0), 0.7, masked2, 0.7, 0)
    return combined

def warp_and_blend_edge_mask_three_photos(fot0, fot1, fot2, H_01, H_21):
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
    blended = (warped0.astype(np.float32)*W0 + canvas1.astype(np.float32)*W1 + warped2.astype(np.float32)*W2)/(W0+W1+W2+1e-6)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return warped0, canvas1, warped2, warped_mask0, canvas_mask1, warped_mask2, blended