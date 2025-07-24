import pathlib
import pkg_resources
from setuptools import find_packages, setup

with pathlib.Path("requirements.txt").open() as f:
    install_requires = [str(requirement) for requirement in pkg_resources.parse_requirements(f)]

setup(
    name="tflops",
    version="0.0.1",
    license="MIT",
    license_files=("LICENSE",),
    description="flood ops",
    URL="https://code.alibaba-inc.com/infer/flops",
    packages=find_packages(),
    install_requires=[],
    python_requires=">=3.8",
)
