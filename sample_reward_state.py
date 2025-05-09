import numpy as np

def sample_goal_and_reward(is_wall, wall_samples, *, bounds=((0., 1.), (0., 1.)), radius_goal=0.1, radius_wall=0.1, rng=None):
    """
    Samples a random goal point in free space and returns a reward function.

    Args:
        is_wall: Callable (x: float, y: float) -> bool
        wall_samples: Callable () -> np.ndarray of shape (N, 2)
        bounds: ((x_min, x_max), (y_min, y_max)) - space limits
        radius_goal: float - radius for +200 reward
        radius_wall: float - radius around wall for -50 penalty
        rng: np.random.Generator or int or None

    Returns:
        reward_fn: Callable(pos: np.ndarray of shape (2,)) -> float
        goal: np.ndarray of shape (2,)
    """
    if isinstance(rng, (int, type(None))):
        rng = np.random.default_rng(rng)

    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]

    # Sample a valid goal point not on a wall
    while True:
        goal = rng.uniform([x_min, y_min], [x_max, y_max])
        if not is_wall(*goal):
            break

    wall_pts = wall_samples()  # (N, 2) wall points

    def reward_fn(pos):
        pos = np.asarray(pos)
        if np.linalg.norm(pos - goal) <= radius_goal:
            return 200.0
        if np.min(np.linalg.norm(wall_pts - pos, axis=1)) < radius_wall:
            return -50.0
        return 0.0

    return reward_fn, goal
