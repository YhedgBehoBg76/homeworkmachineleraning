import random, math, matplotlib.pyplot as plt

eps, min_pts = 50, 3

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def dbscan(pts):
    for p in pts: p += [None, False]
    cid = 0
    for i, p in enumerate(pts):
        if p[3]: continue
        p[3] = True
        neigh = [pts[j] for j in range(len(pts)) if dist(p, pts[j]) <= eps]
        if len(neigh) < min_pts:
            p[2] = -1
        else:
            p[2] = cid
            q = neigh[:]
            k = 0
            while k < len(q):
                n = q[k]
                if not n[3]:
                    n[3] = True
                    nn = [pts[j] for j in range(len(pts)) if dist(n, pts[j]) <= eps]
                    if len(nn) >= min_pts:
                        q += nn
                if n[2] is None:
                    n[2] = cid
                k += 1
            cid += 1
    return pts

# Генерация точек
pts = [[random.randint(0,600), random.randint(0,400)] for _ in range(50)]
pts = dbscan(pts)

# Рисовка
plt.figure(figsize=(6,4))
for x,y,c,_ in pts:
    col = 'black' if c==-1 else plt.cm.tab10(c%10)
    plt.scatter(x, y, c=[col], s=50, marker='o' if c!=-1 else 'x')
plt.xlim(0,600); plt.ylim(0,400); plt.show()
