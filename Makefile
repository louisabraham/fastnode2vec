pypi: dist
	twine upload dist/*
	
dist: clean
	python3 setup.py sdist bdist_wheel

clean:
	-rm -rf *.egg-info build dist

.PHONY: clean
