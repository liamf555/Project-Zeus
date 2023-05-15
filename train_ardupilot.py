import gym
import ardupilot_gym
import wandb
from wandb.integration.sb3 import WandbCallback
from sbx import DroQ
import sys
sys.path.append("/home/dev/mxs") 
# from custom_callbacks import GoToNextCallback
from sb3_contrib.common.wrappers import TimeFeatureWrapper
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from wrappers import wrap_gym


config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 1e6,
    "env_id": "ardupilot_gym/AeropyticsEnvKnife-v0",
}

run = wandb.init(
    project="winp",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    # monitor_gym=True,  # auto-upload the videos of agents playing the game
    # save_code=True,  # optional
)

# learning_starts winp: 1000 sbx: 100
# buffer size? winp: 100000 sbx: 1000000 
#gradient_steps + policy_delay winp: 20
# droput: winp: 0.01  sbx 0.01 or 0.001
# layer norm true
# qf learning rate winp: 3e-4? sbx: 0.001

hyperparams = dict(
learning_starts = 1000,
gradient_steps = 20,
policy_delay = 20,
learning_rate = 0.0003,
buffer_size = 100_000,
qf_learning_rate = 0.0003,
gamma=0.99,
ent_coef= 'auto_0.1',
use_sde = False,
seed=42,
policy_kwargs = dict(layer_norm=True, dropout_rate = 0.01)
)

env = gym.make(config["env_id"])
env = wrap_gym(env)
env.seed(42)
env = make_vec_env(lambda: env, n_envs=1)
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=100)

model = DroQ(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}", **hyperparams)
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=
        WandbCallback(
        model_save_path=f"models/{run.id}",
        verbose=2,
        ),
    
)
run.finish()