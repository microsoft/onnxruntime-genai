'''
Use this script to leverage python to find the stubgen installation.
'''

from mypy.stubgen import main
from argparse import ArgumentParser

def script_main():
    parser = ArgumentParser()
    parser.add_argument('-p', '--package', required=True, help='Package name to generate stubs for')
    parser.add_argument('-o', '--output', required=True, help='Output directory for the generated stubs')
    args = parser.parse_args()

    main(['-p', args.package, '-o', args.output])

if __name__ == '__main__':
    script_main()
