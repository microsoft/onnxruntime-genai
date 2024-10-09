# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
from pathlib import Path


def get_env_var(key):
    return os.environ.get(key)

def generate_nuspec(args):
    lines = ['<?xml version="1.0"?>']
    lines.append("<package>")
    generate_metadata(lines, args)
    generate_files(lines, args)
    lines.append("</package>")
    return lines

def generate_metadata(line_list, args):
    tags = "ONNX;ONNX Runtime;ONNX Runtime Gen AI;Machine Learning"

    metadata_list = ["<metadata>"]
    generate_id(metadata_list, args.package_name)
    generate_version(metadata_list, args.package_version)
    generate_authors(metadata_list, "Microsoft")
    generate_owners(metadata_list, "Microsoft")
    generate_description(metadata_list, args.package_name)
    generate_license(metadata_list)
    generate_readme(metadata_list)
    generate_copyright(metadata_list, "\xc2\xa9 " + "Microsoft Corporation. All rights reserved.")
    generate_project_url(metadata_list, "https://github.com/microsoft/onnxruntime-genai")
    generate_release_notes(metadata_list)
    generate_tags(metadata_list, tags)
    generate_dependencies(metadata_list, args.package_version, args.ort_package_name, args.ort_package_version)
    metadata_list.append("</metadata>")

    line_list += metadata_list

def generate_id(line_list, package_name):
    line_list.append("<id>" + package_name + "</id>")


def generate_version(line_list, package_version):
    line_list.append("<version>" + package_version + "</version>")


def generate_authors(line_list, authors):
    line_list.append("<authors>" + authors + "</authors>")


def generate_owners(line_list, owners):
    line_list.append("<owners>" + owners + "</owners>")


def generate_description(line_list, package_name):
    description = "ONNX Runtime Generative AI Native Package"
    line_list.append("<description>" + description + "</description>")


def generate_copyright(line_list, copyright):
    line_list.append("<copyright>" + copyright + "</copyright>")


def generate_tags(line_list, tags):
    line_list.append("<tags>" + tags + "</tags>")


def generate_icon(line_list, icon_file):
    line_list.append("<icon>" + icon_file + "</icon>")


def generate_license(line_list):
    line_list.append('<license type="file">LICENSE</license>')

def generate_readme(line_list):
    line_list.append('<readme>PACKAGE.md</readme>')

def generate_project_url(line_list, project_url):
    line_list.append("<projectUrl>" + project_url + "</projectUrl>")


def generate_repo_url(line_list, repo_url, commit_id):
    line_list.append('<repository type="git" url="' + repo_url + '"' + ' commit="' + commit_id + '" />')

def generate_release_notes(line_list):
    line_list.append("<releaseNotes>")
    line_list.append("Release Def:")

    branch = get_env_var("BUILD_SOURCEBRANCH")
    line_list.append("\t" + "Branch: " + (branch if branch is not None else ""))

    version = get_env_var("BUILD_SOURCEVERSION")
    line_list.append("\t" + "Commit: " + (version if version is not None else ""))

    line_list.append("</releaseNotes>")

def generate_dependencies(xml_text, package_version, ort_package_name, ort_package_version):
    xml_text.append("<dependencies>")
    target_frameworks = ["NETSTANDARD" , "NETCOREAPP", "NETFRAMEWORK"]
    for framework in target_frameworks:
        xml_text.append(f'<group targetFramework="{framework}">')
        xml_text.append(f'<dependency id="Microsoft.ML.OnnxRuntimeGenAI.Managed" version="{package_version}" />')
        xml_text.append(f'<dependency id="{ort_package_name}" version="{ort_package_version}" />')
        if ort_package_name.endswith("DirectML"):
            xml_text.append(f'<dependency id="Microsoft.AI.DirectML" version="1.15.1" />')
        xml_text.append("</group>")

    xml_text.append("</dependencies>")

def generate_files(lines, args):
    lines.append('<files>')

    lines.append(f'<file src="{args.sources_path}\LICENSE" target="LICENSE" />')
    lines.append(f'<file src="{args.sources_path}\\nuget\PACKAGE.md" target="PACKAGE.md" />')
    lines.append(f'<file src="{args.sources_path}\ThirdPartyNotices.txt" target="ThirdPartyNotices.txt" />')

    def add_native_artifact_if_exists(xml_lines, runtime, artifact):
        p = Path(f"{args.sources_path}/{args.native_build_path}/{runtime}/{args.build_config}/{artifact}")
        if p.exists():
            xml_lines.append(
                f'<file src="{p.absolute()}" target="runtimes\{runtime}\\native" />'
            )

    runtimes = ["win-x64", "win-arm64", "linux-x64", "osx-x64", "osx-arm64"]
    for runtime in runtimes:
      if runtime.startswith("win"):
          add_native_artifact_if_exists(lines, runtime, "onnxruntime-genai.lib")
          add_native_artifact_if_exists(lines, runtime, "onnxruntime-genai.dll")
          add_native_artifact_if_exists(lines, runtime, "d3d12core.dll")
      if runtime.startswith("linux"):
          add_native_artifact_if_exists(lines, runtime, "libonnxruntime-genai.so")
      if runtime.startswith("osx"):
          add_native_artifact_if_exists(lines, runtime, "libonnxruntime-genai.dylib")

    # targets
    for dotnet in ["netstandard2.0", "net8.0", "native"]:
        lines.append(f'<file src="targets\Microsoft.ML.OnnxRuntimeGenAI.targets" target="build\{dotnet}\{args.package_name}.targets" />')
        lines.append(f'<file src="targets\Microsoft.ML.OnnxRuntimeGenAI.props" target="build\{dotnet}\{args.package_name}.props" />')
    # include

    lines.append(f'<file src="{args.sources_path}\src\ort_genai_c.h" target="build\\native\include" />')
    lines.append('</files>')


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ONNX Runtime GenAI create nuget spec script (for hosting native shared library artifacts)",
        usage="",
    )
    # Main arguments
    parser.add_argument("--package_name", required=True, help="ORT GenAI package name")
    parser.add_argument("--package_version", required=True, help="ORT GenAI package version")
    parser.add_argument("--ort_package_name", required=True, help="ORT package name")
    parser.add_argument("--ort_package_version", required=True, help="ORT package version")
    parser.add_argument("--sources_path", required=True, help="OnnxRuntime GenAI source code root.")
    parser.add_argument("--build_config", required=True, help="Eg: RelWithDebInfo")
    parser.add_argument("--native_build_path", required=True, help="Native build output directory.")
    parser.add_argument("--nuspec_output_path", required=True, type=str, help="nuget spec output path.")
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(args)
    # Generate nuspec
    lines = generate_nuspec(args)

    # Create the nuspec needed to generate the Nuget
    print(f"nuspec_output_path: {args.nuspec_output_path}")
    with open(args.nuspec_output_path, "w") as f:
        for line in lines:
            # Uncomment the printing of the line if you need to debug what's produced on a CI machine
            # print(line)
            f.write(line)
            f.write("\n")


if __name__ == '__main__':
    main()
