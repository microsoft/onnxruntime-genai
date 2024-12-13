#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import re
import argparse
from pathlib import Path
from functools import cmp_to_key
import xml.etree.ElementTree as ET

from logger import get_logger

log = get_logger("validate_doxygen")


JAVA_NAMESPACE_PREFIX = "ai::onnxruntime::genai::"
CSHARP_NAMESPACE_PREFIX = "Microsoft::ML::OnnxRuntimeGenAI::"
CXX_NAMESPACE_PREFIX = "Oga"
OBJC_NAMESPACE_PREFIX = "OGA"

OBJC_FUNCTION_SUFFIX_ERROR = ":error:"
OBJC_FUNCTION_SUFFIX_WITH_ERROR = "WithError:"

GETTER_FUNCTION_PREFIX = "get"

IGNORE_CLASSES = ["OgaString", "OgaResult", "OgaHandle"]

# https://github.com/okunishinishi/python-stringcase/blob/master/stringcase.py
def lowercase(string):
    """Convert string into lower case.

    Args:
        string: String to convert.

    Returns:
        string: Lowercase case string.

    """

    return str(string).lower()

def snakecase(string):
    """Convert string into snake case.
    Join punctuation with underscore

    Args:
        string: String to convert.

    Returns:
        string: Snake cased string.

    """

    string = re.sub(r"[\-\.\s]", '_', str(string))
    if not string:
        return string
    return lowercase(string[0]) + re.sub(r"[A-Z]", lambda matched: '_' + lowercase(matched.group(0)), string[1:])


class ParameterMetadata:
    def __init__(self, name):
        self.param_name = name
        self.normalized_name = snakecase(name)


class FunctionMetadata:
    def __init__(self, function_name, params: [ParameterMetadata]):
        self.function_name = function_name
        function_name = function_name.removesuffix(OBJC_FUNCTION_SUFFIX_ERROR)
        function_name = function_name.removesuffix(OBJC_FUNCTION_SUFFIX_WITH_ERROR)

        function_name = function_name.removeprefix(GETTER_FUNCTION_PREFIX)
        parts = function_name.split(':')
        if len(parts) > 1:
            # Assume Objective-C function name. Only check the first part.
            # For example, in the extract name 'decode:length:error:', we only check 'decode'
            function_name = parts[0]
        self.normalized_name = snakecase(function_name)
        self.param = params

class ClassMetadata:

    def __init__(self, class_name, functions: [FunctionMetadata]):
        self.class_name = class_name
        class_name = class_name.removeprefix(CXX_NAMESPACE_PREFIX)
        class_name = class_name.removeprefix(JAVA_NAMESPACE_PREFIX)
        class_name = class_name.removeprefix(CSHARP_NAMESPACE_PREFIX)
        class_name = class_name.removeprefix(OBJC_NAMESPACE_PREFIX)
        self.normalized_name = snakecase(class_name)
        self.functions = functions


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cxx-output-dir')
    parser.add_argument('--java-output-dir')
    parser.add_argument('--csharp-output-dir')
    parser.add_argument('--objc-output-dir')

    return parser.parse_args()

def extract_metadata(output_dir):
    xml_dir = Path(output_dir) / "xml"
    index_xml = xml_dir / "index.xml"

    tree = ET.parse(index_xml)
    root = tree.getroot()
    klasses = root.findall("./compound[@kind='class']")
    structs = root.findall("./compound[@kind='struct']")
    compounds = klasses + structs

    classes = []
    for child in compounds:
        the_class_name = child.findall('./name')[0].text
        if the_class_name in IGNORE_CLASSES:
            continue
        ref_id = child.attrib['refid']

        class_compound_xml_path = xml_dir / f"{ref_id}.xml"
        class_compound_xml_tree = ET.parse(class_compound_xml_path)

        public_funcs = class_compound_xml_tree.findall('./compounddef/sectiondef[@kind="public-func"]/memberdef')

        functions = []

        for f in public_funcs:
            the_function_name = f.findall('./name')[0].text
            params = f.findall('./param/declname')
            parameters = []
            for param in params:
                parameters.append(ParameterMetadata(param.text))
            parameters.sort(key=cmp_to_key(lambda item1, item2: item1.normalized_name > item2.normalized_name))

            functions.append(FunctionMetadata(the_function_name, parameters))

        functions.sort(key=cmp_to_key(lambda item1, item2: item1.normalized_name > item2.normalized_name))

        classes.append(ClassMetadata(the_class_name, functions))

    classes.sort(key=cmp_to_key(lambda item1, item2: item1.normalized_name > item2.normalized_name))

    return classes

def diff_metadata(cxx_metadata: [ClassMetadata], target_metadata: [ClassMetadata]):
    for klass in cxx_metadata:
        if klass.normalized_name == "abstract":
            continue
        target_class = next((x for x in target_metadata if x.normalized_name == klass.normalized_name), None)
        if target_class is None:
            log.warning(f"The class '{klass.class_name}' doesn't exist")
            continue

        #print(jsonpickle.encode(klass, indent=2))
        #print(jsonpickle.encode(target_java_class, indent=2))

        for function in klass.functions:
            target_function = next((x for x in target_class.functions if x.normalized_name == function.normalized_name), None)
            if target_function is None:
                log.warning(f"In class {target_class.class_name}, function '{function.function_name}' doesn't exist")


if __name__ == '__main__':
    args = parse_args()

    cxx_output_dir = args.cxx_output_dir
    java_output_dir = args.java_output_dir
    csharp_output_dir = args.csharp_output_dir
    objc_output_dir = args.objc_output_dir

    cxx_metadata = extract_metadata(cxx_output_dir)
    java_metadata = extract_metadata(java_output_dir)
    csharp_metadata = extract_metadata(csharp_output_dir)
    objc_metadata = extract_metadata(objc_output_dir)

    log.info("/************** Java **************/")
    diff_metadata(cxx_metadata, java_metadata)
    log.info("/************ End Java *************/")

    log.info("/************** C# **************/")
    diff_metadata(cxx_metadata, csharp_metadata)
    log.info("/************ End C# ***************/")

    log.info("/*********** Objective-C **************/")
    diff_metadata(cxx_metadata, objc_metadata)
    log.info("/********** End Objective-C **********/")
