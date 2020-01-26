import numpy as np
from triangulation import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.collections

if __name__ == '__main__':
    num = 10
    radius = 100
    x = np.linspace(1, radius - 1, num)
    y = np.linspace(1, radius - 1, num)
    xv, yx = np.meshgrid(x, y)
    seeds = []
    for i in x:
        for j in y:
            seeds.append([i, j])
    center = np.mean(seeds, axis=0)
    dt = Delaunay(center, 50 * radius)

    for s in seeds:
        dt.add(s)

    print(len(dt.get_triangilation()), "Delaunay triangles")

    fig, ax = plt.subplots()
    ax.margins(0.1)
    ax.set_aspect('equal')
    plt.axis([-1, radius + 1, -1, radius + 1])

    cx, cy = zip(*seeds)
    dt_tris = dt.get_triangilation()
    ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')
    plt.show()