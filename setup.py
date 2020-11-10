from setuptools import setup

def readme():
    with open('README.md', 'r') as f:
        return f.read()

setup(
    name='weno4',
    version='1.1.1',
    description='WENO-4 Interpolation implemented from Janett et al (2019)',
    long_description=readme(),
    long_description_content_type='text/markdown',
    url='http://github.com/Goobley/Weno4Interpolation',
    author='Chris Osborne',
    author_email='c.osborne.1@research.gla.ac.uk',
    license='MIT',
    py_modules=['weno4'],
    install_requires=[
        'numpy',
        'numba'
    ],
    python_requires='>=3.6',
    clasifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License'
    ],
    zip_safe=False
)
