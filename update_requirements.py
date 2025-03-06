import subprocess
import re

def get_latest_version(package_name):
    """
    Get the latest available version of a package from PyPI.
    """
    try:
        output = subprocess.check_output(f"pip index versions {package_name}", stderr=subprocess.STDOUT, shell=True, text=True)
        versions = re.findall(r'\((.*?)\)', output)  # Extract versions from the output
        if versions:
            return versions[0].split(", ")[0]  # Return the highest version found
    except subprocess.CalledProcessError:
        pass
    return None  # If no version is found, return None

# Packages that need fixing
fix_versions = {
    "matplotlib": get_latest_version("matplotlib"),
    "kiwisolver": get_latest_version("kiwisolver"),
}

# Read the current requirements
with open("requirements.txt", "r") as f:
    lines = f.readlines()

# Update the versions
updated_lines = []
for line in lines:
    for package, latest_version in fix_versions.items():
        if package in line and latest_version:
            line = f"{package}=={latest_version}\n"
    updated_lines.append(line)

# Write back the corrected requirements
with open("requirements.txt", "w") as f:
    f.writelines(updated_lines)

print("✅ requirements.txt has been updated with available versions!")
