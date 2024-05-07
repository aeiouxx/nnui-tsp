import os


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def input_folder() -> str:
    return os.path.join(project_root(), 'input')


def input_file(filename: str) -> str:
    return os.path.join(input_folder(), filename)


def logs_folder() -> str:
    return os.path.join(project_root(), 'logs')
