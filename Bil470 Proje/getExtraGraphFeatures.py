def findArticulationPoints(graph):
    numVertices = len(graph)
    disc = [-1] * numVertices
    low = [-1] * numVertices
    visited = [False] * numVertices
    isArticulation = [False] * numVertices
    time = 0

    def DFS(u, parent):
        nonlocal time
        nonlocal graph
        nonlocal disc
        nonlocal low
        nonlocal visited
        nonlocal isArticulation

        childCount = 0
        disc[u] = time
        low[u] = time
        time += 1
        visited[u] = True

        for v in graph[u]:
            if not visited[v]:
                childCount += 1
                DFS(v, u)
                low[u] = min(low[u], low[v])

                if low[v] >= disc[u] and parent is not None:
                    isArticulation[u] = True

            elif v != parent:
                low[u] = min(low[u], disc[v])

        if parent is None and childCount > 1:
            isArticulation[u] = True

    for vertex in range(numVertices):
        if not visited[vertex]:
            DFS(vertex, None)

    count = isArticulation.count(True)
    return count

def calculate_density(graph):
    num_vertices = len(graph)
    num_edges = 0

    for vertex, neighbors in graph.items():
        num_edges += len(neighbors)

    # Divide by 2 to account for undirected edges being counted twice
    num_edges //= 2

    if num_vertices <= 1:
        return 0  # Avoid division by zero or small graphs

    density = 2 * num_edges / (num_vertices * (num_vertices - 1))
    return density