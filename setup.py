from setuptools import setup, find_packages

setup(
    name='ope_methods',  # This is the name of your package
    version='0.1',  # The initial release version
    packages=find_packages(exclude=('tests', 'docs')),  # This tells setuptools to find all packages (subdirectories with an __init__.py file) in the directory
    description='OPE methods',  # A short description of your package
    #long_description=open('README.md').read(),  # A long description, often from a README file
    #long_description_content_type='text/markdown',  # Specifies the format of the long description
    author='Your Name',  # The name of the package author
    author_email='your.email@example.com',  # The email address of the package author
   # url='https://github.com/yourusername/my_package',  # The URL to the repository where your package is hosted
    install_requires=[  # A list of other Python packages that your package depends on
    ],
    classifiers=[
        'Programming Language :: Python :: 3',  # Classifiers help users find your project by categorizing it
        'License :: OSI Approved :: MIT License',  # Again, change the license as needed
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum version requirement of the Python for your package
)