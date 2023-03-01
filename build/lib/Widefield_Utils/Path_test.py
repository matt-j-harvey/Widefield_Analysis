import pathlib
import os

def get_package_file_directory():
    cwd = pathlib.Path.cwd()
    package_root_directory = list(cwd.parts[:-1])
    package_root_directory.append("Files")
    package_file_directory = package_root_directory[0]
    for item in package_root_directory[1:]:
        package_file_directory = os.path.join(package_file_directory, item)

    return package_file_directory