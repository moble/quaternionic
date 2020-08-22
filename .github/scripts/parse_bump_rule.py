import sys


def parse(message):
    if "#prerelease" in message:
        return 'prerelease'

    for pre in ['pre', '']:
        for level in ['patch', 'minor', 'major']:
            if f'#{pre}{level}' in message:
                return f'{pre}{level}'

    return 'patch'


message = sys.stdin.read()
print(parse(message))
