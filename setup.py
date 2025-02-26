from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=["person_detection_robocup", "person_detection_robocup.submodules"],
    package_dir={"": "src"},
)

setup(**d)
