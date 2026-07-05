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
