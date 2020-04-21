pypi: dist
	twine upload dist/*
	
dist:
	-rm dist/*
	python3 setup.py sdist bdist_wheel

clean:
	rm -rf *.egg-info build dist

