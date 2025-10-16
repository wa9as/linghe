rm -rf build && 
rm -rf dist && 
rm -rf linghe.egg-info &&
python setup.py develop && 
python setup.py bdist_wheel && 

# pdoc --output-dir docs -d google --no-include-undocumented --no-search --no-show-source  linghe