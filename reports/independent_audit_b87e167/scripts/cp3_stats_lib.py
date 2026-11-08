import math
import numpy as np
pts = np.array([-math.sqrt(1.5), -1, -math.sqrt(.5), math.sqrt(.5), 1, math.sqrt(1.5)])

def webb(x, months):
    ml = sorted(set(months)); G = len(ml); N = len(x)
    K = np.array([(months == m).sum() for m in ml], float)
    A = np.array([x[months == m].sum() for m in ml])
    W = np.random.Generator(np.random.PCG64(2026111001)).choice(pts, size=(99999, G))
    mu = A.sum() / N; se = math.sqrt(G / (G - 1) * ((A - mu * K) ** 2).sum()) / N; t = mu / se
    As = W * A; mus = As.sum(1) / N
    ses = np.sqrt(G / (G - 1) * ((As - mus[:, None] * K) ** 2).sum(1)) / N
    ts = np.divide(mus, ses, out=np.zeros_like(mus), where=ses > 0)
    return mu, (1 + (ts >= t).sum()) / 100000, G

