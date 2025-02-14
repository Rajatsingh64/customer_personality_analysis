from setuptools import setup, find_packages  # Importing setup tools for package management

# Define the requirements file name
requirements_file = 'requirements.txt'

def get_requirements():
    """
    Function: get_requirements

    Purpose:
    - Reads dependencies from a 'requirements.txt' file.
    - Cleans the list by removing unwanted characters and empty lines.
    - Ignores the '-e .' entry (used for local package installation in editable mode).

    Returns:
    - A cleaned list of requirement package names.
    """

    # Open and read the requirements file
    with open(requirements_file) as require_file:
        requirements_list = require_file.readlines()  # Read all lines into a list

    clean_requirement_list = []  # Initialize an empty list to store cleaned package names
    
    # Process each requirement
    for require_name in requirements_list:
        strip_name = require_name.strip()  # Remove any leading/trailing whitespace
        
        # Ignore the '-e .' entry (used for editable mode in local development)
        if strip_name != "-e .":
            clean_requirement_list.append(strip_name)  # Add cleaned requirement to the list

    return clean_requirement_list  # Return the cleaned requirement list

# Retrieve the cleaned list of dependencies
requirement_list = get_requirements()

# Print the final list of requirements
print(requirement_list)


setup(name="src" ,
      author="Rajat Singh",
      author_email="rajat.k.singh64@gmail.com" ,
      version="0.0.1" ,
      packages=find_packages() ,
      install_requires=get_requirements()
      )