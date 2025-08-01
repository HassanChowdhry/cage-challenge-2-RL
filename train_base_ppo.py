import ray, inspect, gym, torch
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents import RedMeanderAgent, B_lineAgent 

from ray import tune
from ray.rllib.agents.ppo import ppo
from ray.tune.registry import register_env

def create_env(config):
    path = str(inspect.getfile(CybORG))[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env  = ChallengeWrapper(env=cyborg, agent_name='Blue')
    return env
register_env("CybORG-base", lambda config: create_env(config))

ppo_config = ppo.DEFAULT_CONFIG.copy()
ppo_config.update({
    "env": "CybORG-base",            # use our unified environment

    "horizon": 100,                # episode length = 100 steps max
    "framework": "torch",          # use PyTorch (per Mindrake implementation)

    "num_gpus": 1,
    "num_envs_per_worker": 4,

    "model": {
        "fcnet_hiddens": [256, 256, 256, 52],      # two hidden layers of size 256
        "fcnet_activation": "relu",
        "use_lstm": False,
        "use_attention": True,
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
        local_dir="results/base_env",
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
    
    trainer.save("results/base_env/PPO_agent/Best_model")
    trainer.stop()
    
    checkpoint_pointer.close()
    ray.shutdown()