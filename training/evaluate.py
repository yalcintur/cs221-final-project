import numpy as np

def evaluate_rl_model(env, model, num_steps):
    total_profit = 0
    transformer_overloads = 0
    user_satisfaction = []
    total_reward = 0

    obs, _ = env.reset()
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, stats = env.step(action)
        total_reward += reward

        if done:
            if stats:
                print(f"Statistics at step {step}:")
                for key, value in stats.items():
                    print(f"{key}: {value}")

                total_profit += stats.get("total_profits", 0)
                transformer_overloads += stats.get("total_transformer_overload", 0)
                user_satisfaction.append(stats.get("average_user_satisfaction", 0))
            obs, _ = env.reset()

    average_user_satisfaction = np.mean(user_satisfaction) if user_satisfaction else 0
    average_total_profit = total_profit / num_steps

    return {
        "Total Profit": average_total_profit,
        "Transformer Overloads": transformer_overloads,
        "Average User Satisfaction": average_user_satisfaction,
        "Total Reward": total_reward,
    }
