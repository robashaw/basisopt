from setuptools import setup
exec(open('basisopt/version.py').read())

setup(name='basisopt',
      python_requires='>3.9.0',
      version=__version__,
      packages=['basisopt',
                'basisopt.basis',
                'basisopt.wrappers',
                'basisopt.viz',
                'basisopt.opt',
                'basisopt.testing'],
      url='',
      license='MIT',
      author='Robert Shaw',
      author_email='r.shaw@sheffield.ac.uk',
      description='Automatic basis set optimization') 
