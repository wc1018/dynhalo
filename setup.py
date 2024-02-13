import os
import glob
import re
from setuptools import setup, find_packages

project_name = 'dynbahlss'
sources = glob.glob(os.path.join(project_name,'*.py'))

# auto-updating version code taken from RadVel
def get_property(prop, project):
    result = re.search(
        r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
        open(project + "/__init__.py").read(),
    )
    return result.group(1)


def get_requires():
    reqs = []
    for line in open("requirements.txt", "r").readlines():
        reqs.append(line)
    return reqs


dist = setup(
    name=project_name,
    autor="Edgar Salazar",
    author_email="edgarmsc@arizona.edu",
    version=get_property("__version__", project_name),
    description="Modules for reproducing Salazar et. al. (2024) arXiv:####",
    license="MIT License",
    # url="https://github.com/edgarmsalazar/Dynamics-Based-Halo-Model",
    packages=find_packages(),
    package_data={project_name: sources},
    install_requires=get_requires(),
    tests_require=['pytest'],
)