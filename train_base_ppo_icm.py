import ray, inspect, gym, torch
from CybORG import CybORG
from CybORG.Agents.Wrappers import ChallengeWrapper
from CybORG.Agents import RedMeanderAgent, B_lineAgent 

from ray import tune
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import ppo
from curiosity import Curiosity

def create_env(config):
    path = str(inspect.getfile(CybORG))[:-10] + '/Shared/Scenarios/Scenario2.yaml'
    cyborg = CybORG(path, 'sim', agents={'Red': RedMeanderAgent})
    env  = ChallengeWrapper(env=cyborg, agent_name='Blue')
    return env
register_env("CybORG-base", lambda config: create_env(config))

ppo_icm_config = ppo.DEFAULT_CONFIG.copy()
ppo_icm_config.update({
    "env": "CybORG-base",    # our unified env with random attackers
    "num_workers": 0,       # MUST be 0 for Curiosity exploration to avoid parallelism
    "num_gpus": 1,
    "horizon": 100,
    "framework": "torch",
    "model": {
        "fcnet_hiddens": [256, 256, 256],
        "fcnet_activation": "relu",
        "vf_share_layers": False,
    },
    
    "lr": 5e-4,
    "gamma": 0.99,
    "entropy_coeff": 0.001,
    "vf_loss_coeff": 1,  # Scales down the value function loss for better comvergence with PPO
    "clip_param": 0.5,
    "vf_clip_param": 5.0,
    
    "exploration_config": {
        "type": Curiosity,            # use our ICM class
        "framework": "torch",         # ensure torch (the Curiosity class uses torch ops)
        "eta": 1.0,                   # intrinsic reward scale
        "beta": 0.2,                  # forward loss weight
        "lr": 0.001,                  # ICM optimizer learning rate
        "feature_dim": 53,            # dimensionality of state embedding (choose 52 to match obs space)
        "feature_net_config": {       # feature extractor network config
            "fcnet_hiddens": [256], 
            "fcnet_activation": "relu",
            "framework": "torch",
        },
        "inverse_net_hiddens": [256, 256],   # one hidden layer in inverse model
        "inverse_net_activation": "relu",
        "forward_net_hiddens": [256, 256],   # one hidden layer in forward model
        "forward_net_activation": "relu",
        "sub_exploration": { "type": "StochasticSampling" }  # default exploration (policy sampling)
    }
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
        config=ppo_icm_config, 
        stop=stop,
        local_dir="results/base_env", 
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
    
    trainer.save("results/base_env/PPO_ICM_agent/Best_model")
    trainer.stop()
    
    checkpoint_pointer.close()
    ray.shutdown()
