import numpy as np
from PIL import Image
# import cv2
from scipy.signal import convolve2d
from ipdb import set_trace
import time
from datetime import datetime

eps = np.finfo(float).eps


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def ternary_entropyf(pP1, pM1):
    p0 = 1 - pP1 - pM1
    P = np.stack((p0, pP1, pM1))
    H = -(P * np.log2(P))
    H[np.logical_or(P < eps, P > (1 - eps))] = 0
    return np.sum(H)


def calc_lambda(rhoP1, rhoM1, message_length, n):
    l3 = 1e3
    m3 = float(message_length + 1)
    iterations = 0
    while m3 > message_length:
        l3 = l3 * 2

        pP1 = (np.exp(-l3 * rhoP1)) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))
        pM1 = (np.exp(-l3 * rhoM1)) / (1 + np.exp(-l3 * rhoP1) + np.exp(-l3 * rhoM1))

        m3 = ternary_entropyf(pP1, pM1)
        iterations = iterations + 1
        if iterations > 10:
            lam = l3
            return lam
    l1 = 0
    m1 = float(n)
    lam = 0

    alpha = float(message_length) / n
    # limit search to 30 iterations
    while (float(m1 - m3) / n > alpha / 1000.0) and (iterations < 30):
        lam = l1 + (l3 - l1) / 2
        pP1 = (np.exp(-lam * rhoP1)) / (1 + np.exp(-lam * rhoP1) + np.exp(-lam * rhoM1));
        pM1 = (np.exp(-lam * rhoM1)) / (1 + np.exp(-lam * rhoP1) + np.exp(-lam * rhoM1));
        m2 = ternary_entropyf(pP1, pM1)
        if m2 < message_length:
            l3 = lam
            m3 = m2
        else:
            l1 = lam
            m1 = m2
        iterations = iterations + 1

    return lam


def EmbeddingSimulator(x, rhoP1, rhoM1, m, fixEmbeddingChanges=False):
    n = x.size
    lam = calc_lambda(rhoP1, rhoM1, m, n)
    pChangeP1 = (np.exp(-lam * rhoP1)) / (1 + np.exp(-lam * rhoP1) + np.exp(-lam * rhoM1));
    pChangeM1 = (np.exp(-lam * rhoM1)) / (1 + np.exp(-lam * rhoP1) + np.exp(-lam * rhoM1));
    if fixEmbeddingChanges:
        prng = np.random.RandomState(seed=139187)
    else:
        now = datetime.now()
        now_str = now.strftime('%Y %m %d %H %M %S')
        Seed = sum(map(int, now_str.split(' ')))
        prng = np.random.RandomState(Seed)
        
    randChange = prng.random_sample(x.T.shape).T
    y = x.copy()
    y[randChange < pChangeP1] = y[randChange < pChangeP1] + 1
    idx2 = np.logical_and(randChange >= pChangeP1, randChange < pChangeP1 + pChangeM1)
    y[idx2] = y[idx2] - 1

    return y


def S_UNIWARD(cover, payload):
    sgm = 1
    # Get 2D wavelet filters - Daubechies 8
    # 1D high pass decomposition filter

    hpdf = np.asarray(
        [-0.0544158422, 0.3128715909, -0.6756307363, 0.5853546837, 0.0158291053, -0.2840155430,
         -0.0004724846, 0.1287474266, 0.0173693010, -0.0440882539, - 0.0139810279, 0.0087460940,
         0.0048703530, -0.0003917404, -0.0006754494, -0.0001174768],
        dtype=np.float64
    ).reshape(1, -1)  # 1, N
    lpdf = np.ones_like(hpdf, dtype=np.float64)
    lpdf[:, 1::2] = -1
    lpdf = lpdf * hpdf[:, ::-1]

    F1 = lpdf.T * hpdf
    F2 = hpdf.T * lpdf
    F3 = hpdf.T * hpdf

    #cover = cv2.imread(cover_path, flags=cv2.IMREAD_GRAYSCALE).astype(np.float64)
    cover.astype(np.float64)
    wetCost = 10 ** 8

    k, l = cover.shape
    pad_size = max(F1.shape + F2.shape + F3.shape)

    cover_padded = np.pad(cover, ((pad_size, pad_size),), mode='symmetric')

    xi = np.zeros((cover_padded.shape[0], cover_padded.shape[1], 3), dtype=np.float64)
    x = np.zeros((k, l, 3), dtype=np.float64)
    for i in range(3):
        F = locals()['F{}'.format(i + 1)]
        R = conv2(cover_padded, F, mode='same')

        xi[:, :, i] = conv2((1. / (np.abs(R) + sgm)), np.rot90(np.abs(F), 2), mode='same')

        if F.shape[0] % 2 == 0:
            xi[:, :, i] = np.roll(xi[:, :, i], [1, 0], axis=(0, 1))
        if F.shape[1] % 2 == 0:
            xi[:, :, i] = np.roll(xi[:, :, i], [0, 1], axis=(0, 1))

        k_xi, l_xi = xi[:, :, i].shape
        x[:, :, i] = xi[(k_xi - k) // 2: -(k_xi - k) // 2, (l_xi - l) // 2: -(l_xi - l) // 2, i]

    rho = x[:, :, 0] + x[:, :, 1] + x[:, :, 2]

    rho[rho > wetCost] = wetCost
    rho[np.isnan(rho)] = wetCost
    rhoP1 = rho.copy()
    rhoM1 = rho.copy()
    rhoP1[cover == 255] = wetCost
    rhoM1[cover == 0] = wetCost

    # Embedding simulator
    stego = EmbeddingSimulator(cover, rhoP1, rhoM1, payload * cover.size, False)
    return stego.astype(np.uint8)
    #cv2.imwrite('./stego.png', stego)


#print(timeit.timeit("S_UNIWARD('./5355.png', 0.4)", globals=globals(), number=10))

# a = np.random.randint(255, size=(256, 256))


# print(timeit.timeit('np.exp(a)', globals=globals(), number=1000))
# print(timeit.timeit('b = [math.exp(x) for c in a for x in c]', globals=globals(), number=1000))