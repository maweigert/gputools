python setup.py sdist bdist_wheel
python setup.py register -r pypitest
twine upload -r pypitest dist/gputools*
