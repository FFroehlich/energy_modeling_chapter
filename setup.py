from setuptools import setup

setup(name='energy_modeling',
      version='0.0.1',
      description='book chapter about energy modeling in pysb',
      author='',
      packages=[],
      install_requires=[
            'numpy',
            'petab',
            'amici>=0.11.7',
            'pypesto>=0.0.11',
            'pandas',
            'matplotlib',
            'pysb@https://github.com/FFroehlich/pysb@energy_modeling',
      ],
      python_requires='>=3.6',
      zip_safe=False)
