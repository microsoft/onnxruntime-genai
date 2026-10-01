"""
The generated type file is using C++ class names instead of python class names
when used as parameters and return types.
Thus we need to fix it manually here.
"""

import re
from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser()
parser.add_argument("--input", "-i", required=True, help="Input .pyi file")
args = parser.parse_args()

pyi_file = Path(args.input)
content = pyi_file.read_text(encoding="utf-8")
# Only strip the "Oga" prefix at identifier boundaries (word boundary after Oga, before an identifier char)
fixed_content = re.sub(r"\bOga(?=[A-Z])", "", content)
pyi_file.write_text(fixed_content, encoding="utf-8")
