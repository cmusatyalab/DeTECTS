# script to fix compatibility issues for streamlit_elements with newer streamlit versions
import streamlit_elements
import os
import re
def patch_streamlit_elements():

    # issue: https://github.com/okld/streamlit-elements/issues/35
    relative_file_path = 'core/callback.py'
    library_root = list(streamlit_elements.__path__)[0]
    file_path = os.path.join(library_root, relative_file_path)

    # Read broken file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    broken_import = 'from streamlit.components.v1 import components'
    fixed_import = 'from streamlit.components.v1 import custom_component as components\n'

    # Fix broken import line
    for index, line in enumerate(lines):

        if re.match(broken_import, line):
            print(f'Replaced broken import in {file_path}, please restart application.')
            lines[index] = fixed_import

    # Update broken file with fix
    with open(file_path, 'w') as file:
        file.writelines(lines)


if __name__ == "__main__":
    patch_streamlit_elements()