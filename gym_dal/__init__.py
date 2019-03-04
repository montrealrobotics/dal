from gym.envs.registration import register

register(
    id='dal-v0',
    entry_point='gym_dal.envs:DalEnv',
)