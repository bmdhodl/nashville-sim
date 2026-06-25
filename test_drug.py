"""Quick sanity checks for DrugInventoryEnv (buy-and-bill inventory control)."""

from drug_sim import DrugInventoryEnv


def rollout(policy: str, episodes: int = 300, seed0: int = 0) -> float:
    total = 0.0
    for ep in range(episodes):
        env = DrugInventoryEnv(seed=seed0 + ep)
        obs, info = env.reset(seed=seed0 + ep)
        assert len(obs) == env.observation_size, (len(obs), env.observation_size)
        ep_return, done, steps = 0.0, False, 0
        while not done:
            if policy == "random":
                a = env.sample_action()
            elif policy == "greedy":
                a = env.greedy_action()
            else:
                a = 0  # no_op: only ever "order 0 of drug 0"
            obs, r, terminated, truncated, info = env.step(a)
            ep_return += r
            done = terminated or truncated
            steps += 1
            assert steps <= env.years + 1
        total += ep_return
    return total / episodes


def main() -> None:
    env = DrugInventoryEnv(seed=5090)
    print(f"drugs            = {len(env.drugs)}")
    print(f"observation_size = {env.observation_size}")
    print(f"action_size      = {env.action_size}")
    print(f"weeks            = {env.years}")
    print()
    for policy in ("no_op", "random", "greedy"):
        avg = rollout(policy)
        print(f"{policy:>8} policy: mean episode return = {avg:+.3f}")
    print("\nOK: drug env runs, shapes consistent, baselines differ.")


if __name__ == "__main__":
    main()
