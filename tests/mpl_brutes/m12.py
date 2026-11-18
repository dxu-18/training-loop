"""5.12 Saving & exporting: a literal pixel-by-pixel PNG filter pass checked with zlib's own Adler-32, a report laid out
page by page, and palette counting with Python sets."""

import zlib
from collections import Counter

import numpy as np

from . import brute


@brute
def png_row_filters(ns):
    img = ns["img"].tolist()
    H, W = len(img), len(img[0])

    def paeth(a, b, c):
        p = a + b - c
        pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
        if pa <= pb and pa <= pc:
            return a
        return b if pb <= pc else c

    types, total, stream = [], 0, bytearray()
    for r in range(H):
        best = None
        for t in range(5):
            row = []
            for col in range(W):
                x = img[r][col]
                a = img[r][col - 1] if col else 0
                b = img[r - 1][col] if r else 0
                c = img[r - 1][col - 1] if r and col else 0
                pred = (0, a, b, (a + b) // 2, paeth(a, b, c))[t]
                row.append((x - pred) % 256)
            cost = sum(min(v, 256 - v) for v in row)
            if best is None or cost < best[0]:
                best = (cost, t, row)
        types.append(best[1])
        total += best[0]
        stream += bytes([best[1]] + best[2])
    rep = ns["report"]
    if rep == "filters":
        return np.array(types)
    if rep == "cost":
        return total
    return zlib.adler32(bytes(stream))


@brute
def pdf_pagination(ns):
    h, page, gap = ns["h"].tolist(), ns["page"], ns["gap"]

    def layout(s, t):
        pages, used = 1, 0
        for i in range(s, t + 1):
            need = h[i] if used == 0 else used + gap + h[i]
            if need > page:
                pages, used = pages + 1, h[i]
            else:
                used = need
        return pages, used

    rep = ns["report"]
    if rep == "suffix":
        return np.array([layout(s, len(h) - 1)[0] for s in range(len(h))])
    out = [layout(s, t) for s, t in zip(ns["qs"].tolist(), ns["qt"].tolist())]
    return np.array([p if rep == "pages" else u for p, u in out])


@brute
def palette_quantize(ns):
    px = [tuple(p) for p in ns["img"].reshape(-1, 3).tolist()]
    M = ns["M"]
    for k in range(9):
        q = [tuple((v >> k) << k for v in p) for p in px]
        if len(set(q)) <= M:
            break
    rep = ns["report"]
    if rep == "bits":
        return k
    if rep == "palette":
        return len(set(q))
    cnt = Counter(r * 65536 + g * 256 + b for r, g, b in q)
    top = max(cnt.values())
    return min(c for c, v in cnt.items() if v == top)
