import sys


def parse(f):
    import toml
    pyproject = toml.load(f)
    return pyproject['tool']['poetry']['version']


print(parse(sys.argv[1]))
