#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cxx-output-dir')
    parser.add_argument('--java-output-dir')
    parser.add_argument('--csharp-output-dir')

    return parser.parse_args()

def extract_java_metadata(output_dir):
    xml_dir = Path(output_dir) / "xml"
    index_xml = xml_dir / "index.xml"

    print(index_xml)
    tree = ET.parse(index_xml)
    root = tree.getroot()
    klasses = root.findall("./compound[@kind='class']")
    structs = root.findall("./compound[@kind='struct']")
    compounds = klasses + structs
    for child in compounds:
        the_class_name = child.findall('./name')[0].text
        ref_id = child.attrib['refid']
        print(ref_id, the_class_name)

        class_compound_xml_path = xml_dir / f"{ref_id}.xml"
        class_compound_xml_tree = ET.parse(class_compound_xml_path)

        public_funcs = class_compound_xml_tree.findall('./compounddef/sectiondef[@kind="public-func"]/memberdef')

        for f in public_funcs:
            the_function_name = f.findall('./name')[0].text
            params = f.findall('./param/declname')
            print("\t", the_function_name)
            for param in params:
                print("\t\t", param.text)

if __name__ == '__main__':
    args = parse_args()

    cxx_output_dir = args.cxx_output_dir
    java_output_dir = args.java_output_dir
    csharp_output_dir = args.csharp_output_dir

    extract_java_metadata(cxx_output_dir)