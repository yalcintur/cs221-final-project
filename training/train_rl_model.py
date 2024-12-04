from stable_baselines3 import SAC
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.callbacks import EvalCallback
import wandb


def train_rl_model(env, timesteps=5000):
    run = wandb.init(project='ev2gym',
                     sync_tensorboard=True,
                     group='25_cs_V2GProfitPlusLoads',
                     name='ae-sac',
                     save_code=True,
                     )
    
    eval_log_dir = "./eval_logs/"
    
    eval_callback = EvalCallback(env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir,
                                eval_freq=112*50,
                                n_eval_episodes=10, deterministic=True,
                                render=False)
    
    model = SAC("MlpPolicy", env, verbose=1,  device="cuda:0", tensorboard_log="./logs/")
    
    model.learn(total_timesteps=timesteps, progress_bar=True,
                callback=[
                    WandbCallback(
                        gradient_save_freq=100000,
                        model_save_path="models/1",
                        verbose=2),
                    eval_callback])
    
    model.save("sac_model")
    return model