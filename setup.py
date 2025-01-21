from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='tazids',  
    version='1.2.0',  
    description='A simple machine learning library for educational purposes.',
    author='TAZI Mohannad',
    author_email='mohannadtazi.dev@gmail.com',
    url='https://github.com/mohannadtazi/tazi_ds',
    packages=find_packages(),
    install_requires=['numpy'],
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
