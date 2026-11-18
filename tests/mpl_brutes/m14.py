"""5.14 Animation & live-updating plots: real Pillow GIFs, real relim/autoscale_view, a literal bucket simulation."""

import numpy as np

from . import brute


@brute
def gif_timeline(ns):
    import os
    import tempfile
    from PIL import Image
    ids, fps = ns["frame_ids"].tolist(), ns["fps"]
    frames = [Image.new("RGB", (2, 2), ((37 * i) % 256, (91 * i + 40) % 256, (53 * i + 90) % 256)) for i in ids]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "t.gif")
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)  # = PillowWriter
        im = Image.open(path)
        dur = []
        for i in range(im.n_frames):
            im.seek(i)
            dur.append(im.info["duration"])
        im.close()
    if ns["report"] == "durations":
        return np.array(dur)
    if ns["report"] == "longest":
        best = max(dur)
        return [best, dur.index(best)]
    total = sum(dur)
    out = []
    for t in ns["t"].tolist():
        t %= total
        i = 0
        while t >= dur[i]:
            t -= dur[i]
            i += 1
        out.append(i)
    return np.array(out)


@brute
def live_window_autoscale(ns):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loss, W, k = ns["loss"].tolist(), ns["window"], ns["every"]
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    lims = []
    try:
        for i in range(len(loss)):
            if (i + 1) % k == 0:
                a = max(0, i - W + 1)
                line.set_data(range(a, i + 1), loss[a:i + 1])
                ax.relim()
                ax.autoscale_view()
                lims.append(ax.get_ylim())
    finally:
        plt.close(fig)
    lims = np.array(lims)
    if ns["report"] == "ylim":
        return lims
    if ns["report"] == "height":
        return lims[:, 1] - lims[:, 0]
    return int(sum(tuple(lims[j]) != tuple(lims[j - 1]) for j in range(1, len(lims))))


@brute
def live_decimation(ns):
    loss, B = ns["loss"].tolist(), ns["B"]
    cap, buckets = 1, []           # each bucket: list of step indices
    for i in range(len(loss)):
        if buckets and len(buckets[-1]) < cap:
            buckets[-1].append(i)
        else:
            buckets.append([i])
        if len(buckets) > B:
            buckets = [sum(buckets[j:j + 2], []) for j in range(0, len(buckets), 2)]
            cap *= 2
    if ns["report"] == "size":
        return [cap, len(buckets)]
    if ns["report"] == "envelope":
        return np.array([[min(loss[i] for i in b), max(loss[i] for i in b)] for b in buckets])
    pts = []
    for b in buckets:
        vals = [loss[i] for i in b]
        i, j = b[vals.index(min(vals))], b[vals.index(max(vals))]
        for k in sorted({i, j}):
            pts.append((k, loss[k]))
    return np.array(pts)
