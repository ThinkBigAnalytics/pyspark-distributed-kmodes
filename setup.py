import os
from setuptools import setup

from pyspark_kmodes import __version__

setup(name='pyspark_kmodes',
      version = ".".join(map(str, __version__)),
      description='Distributed K-Modes clustering for PySpark',
      classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT',
        'Programming Language :: Python :: 2.7',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis'
      ],
      keywords='k-modes clustering Spark PySpark',
      url='http://github.com/ThinkBigAnalytics/pyspark-distributed-kmodes',
      author='Marissa Saunders',
      author_email='marissa.saunders@thinkbiganalytics.com',
      license='MIT',
      packages=['pyspark_kmodes'],
      install_requires=open('./requirements.txt').read().split(),
      long_description=open('./README.rst').read(),
      include_package_data=True,
      zip_safe=False)
