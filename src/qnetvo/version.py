import pkg_resources

__version__ = pkg_resources.get_distribution(__name__.split(".")[0]).version
