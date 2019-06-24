# -*- coding: utf-8 -*-
"""
This script installs the Anaconda virtual environment required to run calipy.
"""

# System packages
import os
import sys
import shutil
from six.moves import input

def main():
    # Initialize variables
    base_path = None

    # Check that Anaconda is installed and we can run the command 'conda' from shell.
    command_string = 'conda --version'
    print(command_string)
    exit_code = os.system(command_string)
    if exit_code != 0:
        return  # Stop execution. The error message is printed in the console

    # Get the absolute path of this file
    code_dir = os.path.dirname(os.path.realpath(__file__))

    # Check that .yml files exist
    environment_files = list()
    environment_files.append(os.path.join(code_dir, 'calipy.yml'))
    for yml_file in environment_files:
        if not os.path.exists(yml_file):
            raise OSError('%s does not exist.' % yml_file)

    # If we got here, all files exist and conda is installed.


    ############################################################################
    # Update Anaconda's conda
    ############################################################################
    question = 'Update conda? (y/[n])'
    while True:
        # Ask user for instructions
        answer = input(question)
        # Change Enter into the default option ('no')
        if answer == '':
            answer = 'n'
        # Keep only first letter
        answer = answer.lower()[0]
        # Stop if got a 'yes' or 'no'
        if answer in ['y', 'n']:
            break
        else:
            print('Please try again')
            continue

    if answer == 'y':
        # Make a string containing the command to pass to shell
        command_string = 'conda update --yes conda'
        # Run the command
        print(command_string)
        exit_code = os.system(command_string)
        if exit_code != 0:
            return  # Stop execution. The error message is printed in the console

    else:
        print('Will not update conda')

    ############################################################################
    # Current environment
    ############################################################################
    # Get the path from where python is executing
    python_exe_path = sys.executable

    ############################################################################
    # Install environments
    ############################################################################
    for yml_file in environment_files:
        # Read name of environment
        with open(yml_file, 'r') as f:
            txt = f.readlines()
        # Find line to read
        line = [i for i in txt if i.startswith('name:')][0].strip()
        # Remove 'name: ' and keep environment name
        environment_name = line[6:]

        # Check whether environment exists
        env_exists = does_environment_exist(environment_name)
        # If the environment to reinstall is the current one
        if environment_name in python_exe_path:
            print('Cannot re-install the environment from which python is currently running.')
            continue

        # Ask the user what to do
        if env_exists:
            question = 'Would you like to delete and re-install the environment \'%s\'? (y/n)' % environment_name
            while True:
                # Ask user for instructions
                answer = input(question)
                # There is no default option. We need an answer
                if answer == '':
                    print('Please pick one')
                    continue
                # Keep only first letter
                answer = answer.lower()[0]
                # Stop if got a 'yes' or 'no'
                if answer in ['y', 'n']:
                    break
                else:
                    print('Please try again')

        else:  # Otherwise install the environment automatically
            answer = 'y'

        # Process user instructions
        if answer == 'n':
            print('Skipping \'%s\'' % environment_name)
            continue

        if env_exists:
            # Make a string containing the command to pass to Windows' shell
            command_string = 'conda remove -n %s --yes --all' % environment_name
            # Run command
            print(command_string)
            exit_code = os.system(command_string)
            if exit_code != 0:
                return  # Stop execution. The error message is printed in the console

            # Check that folder doesn't exist anymore
            # Get list of environments installed
            list_of_envs = os.popen('conda info --envs').read()
            # Split into lines
            list_of_envs = list_of_envs.split('\n')
            for row in list_of_envs:
                if ' * ' in row:
                    base_path = row.split('*')[1].strip()
                    break
            # If the end path is not 'envs', go down one level
            path_items = base_path.split(os.sep)
            if path_items[-1] != 'envs':
                base_path = os.path.join(base_path, 'envs')
            # Check whether environment folder still exists
            env_dir = os.path.join(base_path, environment_name)
            if os.path.exists(env_dir):  # If so, delete it
                shutil.rmtree(env_dir)

        # Create the environment using the .yml file
        # Make a string containing the command to pass to Windows' shell
        command_string = 'conda env create -n %s -f "%s"' % (environment_name, yml_file)
        # Run command
        print(command_string)
        exit_code = os.system(command_string)
        if exit_code != 0:
            return  # Stop execution. The error message is printed in the console

        # Test installation
        env_exists = does_environment_exist(environment_name)
        if not env_exists:
            raise Exception('Problem in getting access to the environment \'%s\' after creating it. Try reinstalling it' % environment_name)

    print('\nSuccess!!!\n')


################################################################################
# Helper functions
################################################################################
def does_environment_exist(env_name):
    """Check whether the environment exists already.

    :param env_name: [str] The name of the environment to check
    """
    # Make a string containing the command to pass to the shell
    if os.name == 'nt':  # on Windows
        base_command = 'conda activate'
    else:  # on Mac and Linux
        base_command = 'source activate'
    # Append environment name to command
    command_string = '%s %s' % (base_command, env_name)
    # Silence output on Windows
    if os.name == 'nt':
        command_string += ' > NUL 2>&1'

    # Run command
    print(command_string)
    exit_code = os.system(command_string)

    # If shell quit normally the environment exists
    if exit_code == 0:
        return True
    else:  # Otherwise, it doesn't
        return False


################################################################################
# Direct call to function
################################################################################
if __name__ == '__main__':
    main()
