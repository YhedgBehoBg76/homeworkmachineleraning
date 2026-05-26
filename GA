import random
import math

cities = [(0,0), (10,0), (10,8), (0,8)]
n = 4

def distance(i, j):
    return math.hypot(cities[i][0] - cities[j][0], cities[i][1] - cities[j][1])

def route_length(route):
    total = 0
    for i in range(n):
        total += distance(route[i], route[(i + 1) % n])
    return total

population = []
for _ in range(20):
    route = list(range(n))
    random.shuffle(route)
    population.append(route)

for _ in range(100):
    population.sort(key=route_length)
    new_population = []
    for i in range(10):
        new_population.append(population[i])
    for i in range(10):
        parent1 = population[i]
        parent2 = population[i + 1]
        child = parent1[:2]
        for city in parent2:
            if city not in child:
                child.append(city)
        if random.random() < 0.1: #мутация
            i_idx, j_idx = random.sample(range(n), 2)
            child[i_idx], child[j_idx] = child[j_idx], child[i_idx]
        new_population.append(child)
    population = new_population

best_route = population[0]
best_length = route_length(best_route)
print("Маршрут:", best_route, "Длина:", round(best_length, 2))
