from setuptools import find_packages, setup

setup(
    name = 'mmdcritic',
    packages = find_packages(include = 'mmdcritic'),
    version = '0.1.0',
    description = 'Implementation of MMD Critic algorithm based on Scikit Learng framework',
    author = 'Raul De Maio',
    license = 'MIT',
    install_requires=['scikit-learn','numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite = 'tests'
)
