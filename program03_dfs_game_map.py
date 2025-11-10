"""Pretend this graph is a game world with rooms.
Plain DFS walks every room, marking the visit order so we remember what we saw.
Stack version feels close to how class taught it.
"""

world = {
    "Entrance": ["Hall", "Kitchen"],
    "Hall": ["Armory", "Garden"],
    "Kitchen": ["Pantry"],
    "Pantry": [],
    "Armory": ["Boss"],
    "Garden": ["Boss"],
    "Boss": []
}


def dfs_walk(start):
    stack = [(start, [start])]
    visited = []
    seen = set()
    while stack:
        node, path = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        visited.append((node, list(path)))
        for nxt in reversed(world.get(node, [])):
            stack.append((nxt, path + [nxt]))
    return visited


if __name__ == "__main__":
    walk_info = dfs_walk("Entrance")
    for room, path in walk_info:
        print(f"Reached {room} via {path}")
