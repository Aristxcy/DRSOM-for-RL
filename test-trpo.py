import torch

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO
from garage.torch.algos import VPG

# from garage.torch.policies import GaussianMLPPolicy
from policies.gaussian_mlp_policy import GaussianMLPPolicy

from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

@wrap_experiment(log_dir='trpo')
def trpo_pendulum(ctxt=None, seed=1):
    """Train TRPO with InvertedDoublePendulum-v2 environment.
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
    """
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

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                sampler=sampler,
                discount=0.99,
                center_adv=False)

    trainer.setup(algo, env)
    trainer.train(n_epochs=200, batch_size=1024)


for i in range(1):
    trpo_pendulum(seed=1234)