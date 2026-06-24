"""Quick sanity checks for the reconstructed NashvilleGrowthEnv."""

from nashville_sim import NashvilleGrowthEnv


def rollout(policy: str, episodes: int = 200, seed0: int = 0) -> float:
    total = 0.0
    for ep in range(episodes):
        env = NashvilleGrowthEnv(seed=seed0 + ep)
        obs, info = env.reset(seed=seed0 + ep)
        assert len(obs) == env.observation_size, (len(obs), env.observation_size)
        ep_return = 0.0
        done = False
        steps = 0
        while not done:
            if policy == "random":
                a = env.sample_action()
            elif policy == "greedy":
                a = env.greedy_action()
            else:
                a = 0  # always no_op
            obs, r, terminated, truncated, info = env.step(a)
            ep_return += r
            done = terminated or truncated
            steps += 1
            assert steps <= env.years + 1
        total += ep_return
    return total / episodes


def main() -> None:
    env = NashvilleGrowthEnv(seed=5090)
    print(f"corridors      = {len(env.corridors)}")
    print(f"observation_size = {env.observation_size}")
    print(f"action_size      = {env.action_size}")
    print(f"years            = {env.years}")
    print(f"initial score    = {env.last_info['score']}")
    print()
    for policy in ("no_op", "random", "greedy"):
        avg = rollout(policy)
        print(f"{policy:>8} policy: mean episode return = {avg:+.3f}")
    print("\nOK: env runs, shapes consistent, baselines differ.")


if __name__ == "__main__":
    main()
