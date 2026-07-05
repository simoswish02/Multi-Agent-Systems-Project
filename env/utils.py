import numpy as np
from collections import deque


def generate_grid(H: int, W: int, obstacle_density: float, rng: np.random.Generator):
    """
    Generate a random grid of size (H, W) with static obstacles.
    Guarantees:
      - Cell (0,0) is always free (agent start).
      - The target cell is always reachable from (0,0).
    Returns:
      grid: np.ndarray (H, W), 0=free, 1=obstacle
      target_pos: (row, col) tuple
    """
    max_attempts = 200
    for _ in range(max_attempts):
        grid = (rng.random((H, W)) < obstacle_density).astype(np.int32)
        grid[0, 0] = 0  # start always free

        # Collect free cells reachable from (0,0)
        reachable = bfs_reachable(grid, (0, 0))

        # Need at least 2 reachable cells (start + target)
        if len(reachable) < 2:
            continue

        # Choose a target from reachable cells (excluding start)
        candidates = [pos for pos in reachable if pos != (0, 0)]
        idx = int(rng.integers(0, len(candidates)))
        target_pos = candidates[idx]
        return grid, target_pos

    # Fallback: empty grid
    grid = np.zeros((H, W), dtype=np.int32)
    target_pos = (H - 1, W - 1)
    return grid, target_pos


def bfs_reachable(grid: np.ndarray, start: tuple) -> set:
    """
    BFS from start on grid (0=free, 1=obstacle).
    Returns set of reachable (row, col) positions.
    """
    H, W = grid.shape
    visited = set()
    queue = deque([start])
    visited.add(start)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr,nc] == 0 and (nr,nc) not in visited:
                visited.add((nr,nc))
                queue.append((nr,nc))
    return visited


# Maps a unit move to its action index. Keep in sync with grid_env.ACTIONS
# (no import to avoid a circular dependency: grid_env imports env.utils).
_MOVE_TO_ACTION = {(-1, 0): 0, (1, 0): 1, (0, -1): 2, (0, 1): 3}


def bfs_shortest_path(known_obstacles: np.ndarray, start: tuple, goal: tuple):
    """Shortest 4-connected path from ``start`` to ``goal`` on a KNOWN map.

    ``known_obstacles``: (H, W) array where nonzero marks a cell the drone
    knows to be an obstacle. Unknown cells are optimistically assumed free;
    the caller replans every step, so obstacles discovered en route are
    avoided at the next call (the plan can only improve as the map fills in).

    Returns:
        List of cells ``[start, ..., goal]``, or ``None`` when the goal is
        unreachable through cells not known to be obstacles.
    """
    if start == goal:
        return [start]
    H, W = known_obstacles.shape
    parent  = {start: None}
    queue   = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if known_obstacles[nr, nc] or (nr, nc) in parent:
                continue
            parent[(nr, nc)] = (r, c)
            if (nr, nc) == goal:
                path = [(nr, nc)]
                while path[-1] is not None and parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            queue.append((nr, nc))
    return None


def bfs_next_action(known_obstacles: np.ndarray, start: tuple, goal: tuple):
    """First move of the shortest known path as an action index
    (0=up, 1=down, 2=left, 3=right), or ``None`` when no path exists."""
    path = bfs_shortest_path(known_obstacles, start, goal)
    if path is None or len(path) < 2:
        return None
    dr = path[1][0] - path[0][0]
    dc = path[1][1] - path[0][1]
    return _MOVE_TO_ACTION[(dr, dc)]


def auto_nav_action(env, agent_idx: int):
    """Deterministic go-to-target override (eval/simulate only, never training).

    Once the target is present in drone ``agent_idx``'s own known map (seen
    directly or received via comm fusion), return the first move of the BFS
    shortest path to it — planned on that drone's known obstacles — plus the
    full path for GUI display.

    Returns:
        ``(action, path)``; ``(None, None)`` when the drone does not know the
        target yet or no path exists through non-known-obstacle cells (the
        caller falls back to the network action).
    """
    if not env.agent_target[agent_idx].any():
        return None, None
    path = bfs_shortest_path(
        env.agent_obstacle[agent_idx],
        tuple(env.agent_pos[agent_idx]),
        tuple(env.target_pos),
    )
    if path is None or len(path) < 2:
        return None, None
    dr = path[1][0] - path[0][0]
    dc = path[1][1] - path[0][1]
    return _MOVE_TO_ACTION[(dr, dc)], path
