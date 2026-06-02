from setuptools import find_packages, setup
from typing import List
HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''This function will return the list of requirements'''
    requirements = []
    with open(file_path) as file_obj:
        requirments = file_obj.readlines()
        requirements = [req.replace("/n"," ") for req in requirements.txt]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='The project is sample project which helps predict an DDos attack.',
    install_requires = get_requirements('requirements.txt'),
    author='Rohan Sahoo',
    license='',
)
