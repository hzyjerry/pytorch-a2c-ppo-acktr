from setuptools import setup, find_packages

setup(name='ppo',
      packages=find_packages(),
      version='0.1.0',
      # install_requires=['gym', 'matplotlib', 'torch==1.5.0', 'torchvision==0.6.0', 'baselines @ git+git://github.com/openai/baselines.git@master#egg=baselines'])
      install_requires=[
      'gym',
      'matplotlib',
      'tensorflow',
      # 'torch==1.5.0',
      # 'torchvision==0.6.0',
      'torch',
      'torchvision',
      'baselines @ git+https://github.com/openai/baselines.git#egg=baselines'
      ])
