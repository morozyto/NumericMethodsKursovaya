import numpy as np
from triangulation import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri
import matplotlib.collections
import math

def get_square_triang(num=10, radius=100, show=False):
    x = np.linspace(0, radius, num)
    y = np.linspace(0, radius, num)
    xv, yx = np.meshgrid(x, y)
    seeds = []
    for i in x:
        for j in y:
            seeds.append([i, j])
    center = np.mean(seeds, axis=0)
    dt = Delaunay(center, 50 * radius)

    for s in seeds:
        dt.add(s)

    if show:
        fig, ax = plt.subplots()
        ax.margins(0.1)
        ax.set_aspect('equal')
        plt.axis([-1, radius + 1, -1, radius + 1])

        cx, cy = zip(*seeds)
        dt_tris = dt.get_triangilation()
        ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')
        plt.show()

    return seeds, dt.get_triangilation(), []

def is_in_circle(center_x, center_y, r, x, y):
    return (x - center_x) ** 2 + (y - center_y) ** 2 < r ** 2


def get_square_circled_triang(num=10, radius=100, show=False):
    x = np.linspace(0, radius, num)
    y = np.linspace(0, radius, num)
    seeds = []
    for i in x:
        for j in y:
            seeds.append([i, j])

    scape_radius = radius / num

    circle_radius = radius / 10
    circle_x = radius / 2
    circle_y = 0

    seeds = [ point for point in seeds if not is_in_circle(circle_x, circle_y, circle_radius, point[0], point[1])]

    n = max(3, int(circle_radius * math.pi / scape_radius))
    step_angle = math.pi / n

    bad_triangle_points = set()

    for i in range(n + 1):
        bad_triangle_points.add(len(seeds))
        seeds.append([ circle_x + circle_radius*math.cos(step_angle * i), circle_y + circle_radius*math.sin(step_angle * i)])

    center = np.mean(seeds, axis=0)

    dt = Delaunay(center, 50 * radius)

    for s in seeds:
        dt.add(s)

    res = dt.get_triangilation()

    def check_triangle(obj):
        return (obj[0] in bad_triangle_points) + (obj[1] in bad_triangle_points) + (obj[2] in bad_triangle_points)

    def check_triangle2(obj):
        return (not is_in_circle(circle_x, circle_y, circle_radius, seeds[obj[0]][0], seeds[obj[0]][1]) or
                     not is_in_circle(circle_x, circle_y, circle_radius, seeds[obj[1]][0], seeds[obj[1]][1]) or
                     not is_in_circle(circle_x, circle_y, circle_radius, seeds[obj[2]][0], seeds[obj[2]][1]))

    res = [item for item in res if (check_triangle(item) == 2 and check_triangle2(item)) or check_triangle(item) < 2]

    if show:
        fig, ax = plt.subplots()
        ax.margins(0.1)
        ax.set_aspect('equal')
        plt.axis([-1, radius + 1, -1, radius + 1])

        cx, cy = zip(*seeds)
        dt_tris = res
        ax.triplot(matplotlib.tri.Triangulation(cx, cy, dt_tris), 'bo--')
        plt.show()

    return seeds, res, bad_triangle_points

if __name__ == '__main__':
    # _, showed_value, _ = get_square_triang(show=True)
    seeds, showed_value, t = get_square_circled_triang(num=10, show=True)
    print(seeds)
    print(showed_value)
    print(t)

