import setuptools

setuptools.setup(
    name="DiamondMiner",
    version="0.0.3",
    url="https://github.com/mainCSG/diamondminer",
    packages=setuptools.find_packages(),
    install_requires=[
        "opencv-python",
        "matplotlib",
        "numpy",
        "pandas",
        "PyQt5",
        "scipy"
    ],
    author="Andrija Paurevic",
    author_email="apaurevic@uwaterloo.ca",

    description="Extract Coulomb diamond information from measurements",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
)