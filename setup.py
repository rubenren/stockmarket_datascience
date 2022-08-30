from distutils.core import setup


def readme():
  """Import the README.md Markdown file and try to convert it to RST format."""
    try:
    import pypandoc
    return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
      with open('README.md') as readme_file:
        return readme_file.read()


        setup(
            name='stockmarket',
            version='0.1',
            description='Analysis of the Stock market',
            long_description=readme(),
            classifiers=[
              'Programming Language :: Python :: 3',
            ],
            url='https://github.com/rubenren/stockmarket_datascience',
            author='Ruben Renteria',
            author_email='renteria.c.ruben@gmail.com',
            packages=['stockmarket'],
            install_requires=[
                'pypandoc>=1.8.1'
            ]
        )
