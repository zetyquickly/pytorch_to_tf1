from setuptools import setup, find_namespace_packages, find_packages

print(find_namespace_packages())
setup(name='pytorch_to_tf1', version='1.0', packages=find_packages())