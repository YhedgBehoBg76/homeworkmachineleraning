import pygame
import random
import math

#параметры
eps = 50
min_pts = 3

class Point:
    def init(self, x, y):  # конструктор
        self.x = x
        self.y = y
        self.cluster = None
        self.visited = False

def generate_colors(n: int):
    colors = []
    for _ in range(n):
        colors.append((
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
    colors.append((255, 0, 0))  # цвет для шума
    return colors

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)  2 + (p1.y - p2.y)  2)

#поиск соседей
def region_query(points, point):
    neighbors = []
    for p in points:
        if distance(point, p) <= eps:
            neighbors.append(p)
    return neighbors

def expand_cluster(points, point, neighbors, cluster_id):
    point.cluster = cluster_id
    for i in range(len(neighbors)):
        neighbor = neighbors[i]

        if not neighbor.visited:
            neighbor.visited = True
            new_neighbors = region_query(points, neighbor)

            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors

        if neighbor.cluster is None:
            neighbor.cluster = cluster_id

def my_dbscan(points):
    cluster_id = 0

    for p in points:
        p.visited = False
        p.cluster = None

    for point in points:
        if point.visited:
            continue

        point.visited = True
        neighbors = region_query(points, point)

        if len(neighbors) < min_pts:
            point.cluster = -1  # шум
        else:
            expand_cluster(points, point, neighbors, cluster_id)
            cluster_id += 1

def my_pygame():
    points = []
    pygame.init()
    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    pygame.display.set_caption("DBSCAN")

    running = True
    while running:
        screen.fill((255, 255, 255))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            #перерисовка при изменении окна
            if event.type == pygame.WINDOWRESIZED:
                screen.fill((255, 255, 255))

            #добавление точки
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    points.append(Point(*event.pos))

            #запуск DBSCAN
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    my_dbscan(points)

        clusters = [p.cluster for p in points if p.cluster is not None and p.cluster != -1]
        max_cluster = max(clusters) + 1 if clusters else 0
        colors = generate_colors(max_cluster)

        #отрисовка точек
        for p in points:
            if p.cluster == -1:
                color = (255, 0, 0)  # шум
            else:
                color = colors[p.cluster]

            size = 5  
            pygame.draw.line(screen, color, (p.x - size, p.y), (p.x + size, p.y), 2)
            pygame.draw.line(screen, color, (p.x, p.y - size), (p.x, p.y + size), 2)

        pygame.display.flip()

    pygame.quit()

def main():
    my_pygame()

if name == "main":
    main()
