import pandas as pd
import json
import glob
import re
from utils.parse_tools import chunk_strings

def main():
    subsections = {}

    # get all txt files in the peps directory
    files = glob.glob('peps/*.txt')

    
    for file in files:
        # Get name of pep file : peps/pep-0503.txt -> pep-0503
        name = file.split('/')[1].split('.')[0]
        # Read file
        with open(file, 'r') as f:
            lines = f.read()

        # extract status from content
        pattern = r'\nStatus:\s(.*?)\n'
        match = re.search(pattern, lines)
        if match:
            status = match.group(1)
        else:
            status = 'unknown'

        # do not include if pep is rejected
        if status == 'Rejected':
            continue

        if 'Status: Rejected' in lines:
            continue

        # extract title from content
        pattern = r'\nTitle:\s(.*?)\n'
        match = re.search(pattern, lines)
        if match:
            title = match.group(1)
        else:
            title = 'unknown'


        # trim lines so that it ends at '\n\n\nReferences\n==========\n\n'
        pattern = r'\nReferences\n{1,}[=-]{2,}'
        match = re.search(pattern, lines)
        if match:
            lines = lines[:match.start()]
        else:
            lines = lines

        # trim lines so that it ends at 'Copyright'
        pattern = r'\nCopyright\n={2,}\n'
        match = re.search(pattern, lines)
        if match:
            lines = lines[:match.start()]
        else:
            lines = lines

        # Split lines into reasonably sized sections
        line_sections = re.split(r'(?=\n\n\n[\w\s]+?\n={2,}\n\n)', lines)
        line_sections = chunk_strings(line_sections, 2000)

        for idx, section in enumerate(line_sections):

            subsections[f'{name}_{idx}'] = {
                'title': title,
                'status': status,
                'content': section
                }

   # save to json file
    with open('resources/peps.json', 'w') as f:
        json.dump(subsections, f, indent=4)


if __name__ == '__main__':
    main()