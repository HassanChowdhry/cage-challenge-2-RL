import ray
from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import ppo
from curiosity import Curiosity
from envs.unified_cyborg_env import UnifiedCybORGEnv

register_env("CybORG-uni", lambda config: UnifiedCybORGEnv(config))

ppo_icm_config = ppo.DEFAULT_CONFIG.copy()
ppo_icm_config.update({
    "env": "CybORG-uni",    # our unified env with random attackers
    "num_workers": 0,       # MUST be 0 for Curiosity exploration to avoid parallelism issues:contentReference[oaicite:31]{index=31}
    "horizon": 100,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [256, 256],
        "fcnet_activation": "relu",
    },
    "lr": 5e-4,
    "gamma": 0.99,
    "entropy_coeff": 0.001,
    "exploration_config": {
        "type": Curiosity,            # use our ICM class
        "framework": "torch",         # ensure torch (the Curiosity class uses torch ops)
        "eta": 1.0,                   # intrinsic reward scale
        "beta": 0.2,                  # forward loss weight
        "lr": 0.001,                  # ICM optimizer learning rate
        "feature_dim": 52,            # dimensionality of state embedding (choose 52 to match obs space)
        "feature_net_config": {       # feature extractor network config
            "fcnet_hiddens": [], 
            "fcnet_activation": "relu",
            "framework": "torch",
        },
        "inverse_net_hiddens": [256],   # one hidden layer in inverse model
        "inverse_net_activation": "relu",
        "forward_net_hiddens": [256],   # one hidden layer in forward model
        "forward_net_activation": "relu",
        "sub_exploration": { "type": "StochasticSampling" }  # default exploration (policy sampling)
    }
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
        config=ppo_icm_config, 
        stop=stop,
        local_dir="results/unified_env", 
        name="PPO_ICM_agent",
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
    
    trainer = algo(config=ppo_icm_config)
    trainer.restore(last_checkpoint)
    print(trainer.get_policy().get_weights())
    
    trainer.save("results/unified_env/PPO_ICM_agent/Best_model")
    trainer.stop()
    
    checkpoint_pointer.close()
    ray.shutdown()
