import setuptools

setuptools.setup(
    name="CeSoR",
    version="0.0.1",
    author="Ido Greenberg",
    description="Cross-entropy Soft-Risk optimization algorithm",
    url="https://github.com/ido90/CrossEntropySampler",
    packages=setuptools.find_packages(),
    install_requires = ["numpy","scipy","pandas","torch"])