from setuptools import find_packages, setup

setup(
    author="F. Dangel",
    name="backobs-for-pytorch",
    version="0.0.1",
    description=r"Use BackPACK on DeepOBS problems",
    url="https://github.com/f-dangel/backobs",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "backpack-for-pytorch>=1.1.1",
        "deepobs>=1.2.0b",
    ],
    zip_safe=False,
    python_requires=">=3.7",
)
