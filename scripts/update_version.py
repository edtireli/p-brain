def update_version(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('"')
            major, minor, patch = map(int, version.split('.'))
            new_patch = patch + 1
            new_version = f"{major}.{minor}.{new_patch}"
            lines[i] = f'__version__ = "{new_version}"\n'
            break

    with open(file_path, 'w') as f:
        f.writelines(lines)

    return new_version

if __name__ == "__main__":
    new_version = update_version('modules/__init__.py')
