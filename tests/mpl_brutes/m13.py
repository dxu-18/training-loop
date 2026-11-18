"""5.13 Patches, collections & coordinate systems: Matplotlib's own geometry (Path, Bbox, transforms) on small inputs."""

import numpy as np

from . import brute


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


@brute
def lasso_selection(ns):
    from matplotlib.path import Path
    inside = Path(ns["poly"]).contains_points(ns["pts"])
    if ns["report"] == "mask":
        return inside
    if ns["report"] == "count":
        return int(inside.sum())
    out = [0] * ns["k"]
    for lab, ok in zip(ns["labels"].tolist(), inside.tolist()):
        out[lab] += ok
    return np.array(out)


@brute
def detection_nms(ns):
    from matplotlib.transforms import Bbox
    boxes, gt, scores, thr = ns["boxes"].tolist(), ns["gt"].tolist(), ns["scores"].tolist(), ns["thr"]

    def iou(a, b):
        A, B = Bbox.from_bounds(*a), Bbox.from_bounds(*b)
        inter = Bbox.intersection(A, B)
        i = 0 if inter is None else inter.width * inter.height
        return i / (A.width * A.height + B.width * B.height - i)

    order = sorted(range(len(boxes)), key=lambda i: (-scores[i], i))
    if ns["report"] == "nms":
        keep, dead = [], set()
        for i in order:
            if i in dead:
                continue
            keep.append(i)
            dead |= {j for j in range(len(boxes)) if iou(boxes[i], boxes[j]) > thr}
        return keep
    if ns["report"] == "max_iou":
        return np.array([max(iou(b, q) for q in gt) for b in boxes])
    used, tp = set(), 0
    for i in order:
        best, bj = -1.0, -1
        for j, q in enumerate(gt):
            if j not in used and iou(boxes[i], q) > best:
                best, bj = iou(boxes[i], q), j
        if bj >= 0 and best >= thr:
            used.add(bj)
            tp += 1
    return tp


@brute
def box_union_area(ns):
    r = ns["rects"]
    side = int((r[:, 0] + r[:, 2]).max()) + 1, int((r[:, 1] + r[:, 3]).max()) + 1
    cover = np.zeros(side, dtype=int)            # unit cells of the integer grid
    for x, y, w, h in r.tolist():
        cover[x:x + w, y:y + h] += 1
    return int({"union": (cover >= 1), "overlap": (cover >= 2), "exactly_one": (cover == 1)}[ns["report"]].sum())


@brute
def label_declutter(ns):
    plt = _plt()
    fig = plt.figure(figsize=ns["figsize"], dpi=ns["dpi"])
    try:
        ax = fig.add_axes(ns["position"])
        ax.set_xscale(ns["xscale"])
        ax.set_xlim(*ns["xlim"])
        ax.set_ylim(*ns["ylim"])
        pix = ax.transData.transform(ns["pts"])
    finally:
        plt.close(fig)
    if ns["report"] == "pixels":
        return pix
    bw, bh = ns["box"]
    score = ns["score"].tolist()
    placed = []
    for i in sorted(range(ns["n"]), key=lambda i: (-score[i], i)):
        if all(not (abs(pix[i, 0] - pix[j, 0]) < bw and abs(pix[i, 1] - pix[j, 1]) < bh) for j in placed):
            placed.append(i)
    return placed if ns["report"] == "placed" else ns["n"] - len(placed)
