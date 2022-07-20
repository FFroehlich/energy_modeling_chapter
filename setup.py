from setuptools import setup

setup(name='energy_modeling',
      version='1.0.0',
      description='book chapter about energy modeling in pysb',
      author='',
      packages=[],
      install_requires=[
            'numpy',
            'petab==0.1.27',
            'amici==0.11.32',
            'pypesto==0.2.13',
            'fides==0.7.5',
            'pandas',
            'matplotlib',
            'pysb@https://github.com/FFroehlich/pysb@energy_modeling',
      ],
      python_requires='>=3.8',
      zip_safe=False)
