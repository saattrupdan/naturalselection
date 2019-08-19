import setuptools

with open("README.md", "r") as file_in:
    long_description = file_in.read()

setuptools.setup(
    name = 'saattrupdan.darwin',  
    version = '0.1',
    author = "Dan Saattrup Nielsen",
    author_email = "saattrupdan@gmail.com",
    description = "An all-purpose pythonic genetic algorithm",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/saattrupdan/darwin",
    packages = setuptools.find_packages(),
    classifiers = [
     "Programming Language :: Python :: 3",
     "License :: OSI Approved :: MIT License",
     "Operating System :: OS Independent",
    ],
    )
