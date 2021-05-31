import findspark
from os import environ, getcwd
from pathlib import Path
findspark.init()

environ['PYSPARK_PYTHON'] = f'{Path(getcwd())}/.tox/dev/bin/python'
