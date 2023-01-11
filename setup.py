from setuptools import setup, find_namespace_packages
exec(
    open('basisopt/version.py', 'r',
         encoding='utf-8').read()
)

setup(name='basisopt',
      python_requires='>3.9.0',
      version=__version__,
      packages=find_namespace_packages(),
      license='MIT',
      author='Robert Shaw',
      author_email='robertshaw383@gmail.com',
      description='Automatic basis set optimization for quantum chemistry',
      install_requires=[
          "colorlog",
          "numpy",
          "scipy",
          "pandas",
          "matplotlib",
          "monty",
          "basis_set_exchange >= 0.9",
          "mendeleev == 0.9.0"
      ],
      extras_require={
          "test": [
              "pytest",
              "pytest-cov",
          ]
      },
) 
