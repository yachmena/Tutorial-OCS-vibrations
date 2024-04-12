import subprocess
import sys


def install_packages(requirements):
    with open(requirements, "r") as f:
        packages = f.readlines()
        packages = [package.strip() for package in packages]

    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])


if __name__ == "__main__":
    requirements = "requirements.txt"
    install_packages(requirements)
