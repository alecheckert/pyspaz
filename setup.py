'''
setup.py
'''
import setuptools

setuptools.setup(
	name = "pyspaz",
	version = "1.0",
	packages = setuptools.find_packages(),
	install_requires = [
		'nd2reader==3.0.9',
		'tifffile>=0.14.0',
		'munkres>=1.1.2',
	],
)
	
