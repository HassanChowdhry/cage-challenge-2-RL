import ray, torch

from ray import tune
from ray.rllib.agents.ppo import ppo
from ray.tune.registry import register_env

from envs.unified_cyborg_env import UnifiedCybORGEnv

register_env("CybORG-uni", lambda config: UnifiedCybORGEnv(config))

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update({
    "env": "CybORG-uni",            # use our unified environment

    "horizon": 100,                # episode length = 100 steps max
    "framework": "torch",          # use PyTorch (per Mindrake implementation)

    "num_gpus": 1,
    "num_envs_per_worker": 4,

    "model": {
        "fcnet_hiddens": [256, 256, 256, 52],      # two hidden layers of size 256
        "fcnet_activation": "relu",
        "use_lstm": False,
	"use_attention": False,
	"lstm_use_prev_action": True,
	"lstm_use_prev_reward": True,
    },
    "lr": 5e-4,                    # learning rate (could tune; Mindrake used 5e-4 for B-Line
    "gamma": 0.99,
    "entropy_coeff": 0.001,        # encourage exploration (if needed)
    "rollout_fragment_length": 100,
    "num_sgd_iter": 10,
})


if __name__ == "__main__":
    ray.init(
        num_gpus=1,
        local_mode=False,
    )
    print("torch.cuda.is_available()", torch.cuda.is_available())
    torch.device(str("cuda:0"))

    stop = {
        "training_iteration": 10_000_000,   # The number of times tune.report() has been called
        "timesteps_total": 100_000_000,   # Total number of timesteps
	#"timesteps_total": 100,
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
    # print(trainer.get_policy().get_weights())

    trainer.save("results/unified_env/PPO_agent/Best_model")
    trainer.stop()

    checkpoint_pointer.close()

    ray.shutdown()
