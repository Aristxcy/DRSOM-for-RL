import torch
import gym
from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO

# from garage.torch.policies import GaussianMLPPolicy
from policies.gaussian_mlp_policy import GaussianMLPPolicy

from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

from TRPO_DRSOM import TRPO_DRSOM

@wrap_experiment(log_dir='drsom_test/g_linesearch')
def my_trpo(ctxt=None, seed=1):
    set_seed(seed)

    env = GymEnv('MountainCarContinuous-v0')

    trainer = Trainer(ctxt)
    policy = GaussianMLPPolicy(env.spec,
                               hidden_sizes=[32, 32],
                               hidden_nonlinearity=torch.tanh,
                               output_nonlinearity=None)
    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)
    algo = TRPO_DRSOM(env_spec=env.spec,
                      policy=policy,
                      value_function=value_function,
                      sampler=sampler,
                      discount=0.99,
                      center_adv=False)
    trainer.setup(algo, env)
    trainer.train(n_epochs=500, batch_size=1024)

for i in range(1):
    my_trpo(seed=1234)