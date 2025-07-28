import ray

from ray import tune
from ray.rllib.agents.ppo import ppo
from ray.tune.registry import register_env

from envs.unified_cyborg_env import UnifiedCybORGEnv

register_env("CybORG-uni", lambda config: UnifiedCybORGEnv(config))

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update({
    "env": "CybORG-uni",            # use our unified environment
    "num_workers": 4,              # parallel workers (adjust based on CPU cores; use 0 if curiosity is added)
    "num_envs_per_worker": 1,      # or more for vectorization; can use >1 to parallelize env sampling
    "horizon": 100,                # episode length = 100 steps max
    "framework": "torch",          # use PyTorch (per Mindrake implementation)
    "model": {
        "fcnet_hiddens": [256, 256],      # two hidden layers of size 256
        "fcnet_activation": "relu",
    },
    "lr": 5e-4,                    # learning rate (could tune; Mindrake used 5e-4 for B-Line
    "gamma": 0.99,
    "entropy_coeff": 0.001,        # encourage exploration (if needed)
})


if __name__ == "__main__":
    ray.init()

    stop = {
        "training_iteration": 10000000,   # The number of times tune.report() has been called
        "timesteps_total": 100000000,   # Total number of timesteps
        "episode_reward_mean": -0.1,
    }
    
    algo = ppo.PPOTrainer
    results = tune.run(
        algo, 
        config=ppo_config, 
        stop=stop,               
        local_dir="results/unified_env",
        name="PPO_agent",
        checkpoint_at_end=True,
        checkpoint_freq=1,
        keep_checkpoints_num=3,
        checkpoint_score_attr="episode_reward_mean",
    )
    
    checkpoint_pointer = open("checkpoint_pointer.txt", "w")
    last_checkpoint = results.get_last_checkpoint(
        metric="episode_reward_mean", mode="max"
    )

    checkpoint_pointer.write(str(last_checkpoint))
    print("Best model checkpoint written to: {}".format(last_checkpoint))
    
    trainer = algo(config=ppo_config)
    trainer.restore(last_checkpoint)
    print(trainer.get_policy().get_weights())
    
    trainer.save("results/unified_env/PPO_agent/Best_model")
    trainer.stop()
    
    checkpoint_pointer.close()
    
    ray.shutdown()