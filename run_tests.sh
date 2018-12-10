# runs pytest on all tests for both python2 and python3

python2 -m pytest -p no:warnings -v tests/
python3 -m pytest -v tests/
