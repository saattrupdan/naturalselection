import setuptools

with open("README.md", "r") as file_in:
    long_description = file_in.read()

setuptools.setup(
    name = 'naturalselection',  
    entry_points = {'console_scripts' : 
        ['core = naturalselection.core', 'nn = naturalselection.nn']},
    install_requires = ['numpy', 'functools', 'matplotlib.pyplot', 'tqdm',
                        'multiprocessing', 'contextlib', 'time', 'itertools',
                        'tensorflow', 'sklearn', 'os', 'sys'],
    version = "0.1",
    author = "Dan Saattrup Nielsen",
    author_email = "saattrupdan@gmail.com",
    description = "An all-purpose pythonic genetic algorithm",
    keywords = "genetic algorithm neural network",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/saattrupdan/naturalselection",
    packages = setuptools.find_packages(),
    classifiers = [
     "Programming Language :: Python :: 3",
     "License :: OSI Approved :: Python Software Foundation License",
     "Operating System :: OS Independent",
    ],
    )
