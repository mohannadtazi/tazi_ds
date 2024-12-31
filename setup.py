from setuptools import setup, find_packages

setup(
    name='tazi_ds',  # Name of your library
    version='1.0.0',
    description='A simple linear regression library',
    author='TAZI Mohannad',
    author_email='mohannadtazi.dev@gmail.com',
    url='https://github.com/mohannadtazi/tazi_ds',  # Replace with your repository
    packages=find_packages(),
    install_requires=['numpy'],  # Dependencies
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Minimum Python version
)
