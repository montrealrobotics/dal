from setuptools import setup, find_packages

setup(name='a2c-ppo-acktr',
      packages=find_packages(),
      version='0.0.1',
      install_requires=['gym', 'matplotlib', 'pybullet'])

setup(
    name='gym_dal',
    version='0.0.1',
    # keywords='memory, environment, agent, rl, openaigym, openai-gym, gym',
    install_requires=[
        'gym>=0.9.6',
        'numpy>=1.10.0',
        #'pyqt5>=5.10.1'
    ]
)
