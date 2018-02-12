from setuptools import setup, find_packages

setup(name='model_train',
  version='0.1',
  packages=find_packages(),
  description='example to run keras on gcloud ml-engine',
  install_requires=[
      'keras',
      'h5py'
  ],
  zip_safe=False)