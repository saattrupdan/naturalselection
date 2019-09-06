import setuptools

with open("README.md", "r") as file_in:
    long_description = file_in.read()

setuptools.setup(
    name = 'naturalselection',  
    entry_points = {'console_scripts' : [
        'core = naturalselection.core:main', 
        'nn = naturalselection.nn:main', 
        ]},
    install_requires = ['numpy','matplotlib','tqdm','tensorflow','sklearn'],
    version = "0.6.0",
    author = "Dan Saattrup Nielsen",
    author_email = "saattrupdan@gmail.com",
    description = "An all-purpose pythonic genetic algorithm",
    keywords = "genetic algorithm neural network",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/saattrupdan/naturalselection",
    packages = setuptools.find_packages(),
    classifiers = [
     "Development Status :: 3 - Alpha",
     "Programming Language :: Python :: 3",
     "License :: OSI Approved :: MIT License",
     "Operating System :: OS Independent",
    ],
    )
