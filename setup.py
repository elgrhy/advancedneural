from setuptools import setup, find_packages

setup(
    name='advancedneural',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=1.7.0',  # Adjust the version based on your requirements
    ],
    entry_points={
        'console_scripts': [],
    },
)
