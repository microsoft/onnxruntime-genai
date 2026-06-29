"""
The generated type file is using C++ class names instead of python class names 
when used as parameters and return types.
Thus we need to fix it manually here.
"""

from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="Input .pyi file")
args = parser.parse_args()

pyi_file = Path(args.input)
content = pyi_file.read_text()
fixed_content = content.replace("Oga", "")
pyi_file.write_text(fixed_content)
