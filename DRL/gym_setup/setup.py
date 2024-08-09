from setuptools import setup, find_packages


"""describe path names and library dependencies required for installation"""

setup(
    name="AAE_DRL_env",
    version="0.0.1",
    install_requires=["gymnasium"],
    packages=find_packages(include=["DRL", "DRL.*"]),
)


#python setup.py develop : edit without reinstalling
#python setup.py clean : clean tmp files
# python setup.py bdist_wheel : wheel