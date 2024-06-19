#!/usr/bin/env python3
"""
Sphinx documentation build script
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Iterable
from glob import glob
from typing import Any
from weakref import ReferenceType

from make_util import get_package_version
from packaging import version as pkg_version

DIR_SPHINX_ROOT = os.path.dirname(os.path.dirname(__file__))
assert (
    os.path.basename(DIR_SPHINX_ROOT) == "sphinx"
), f"Sphinx root dir {DIR_SPHINX_ROOT!r} is named 'sphinx'"

# Sphinx commands
CMD_SPHINX_BUILD = "sphinx-build"
CMD_SPHINX_AUTOGEN = "sphinx-autogen"

# File paths
DIR_SPHINX_BASE = os.path.dirname(os.path.realpath(__file__))
DIR_REPO_ROOT = os.path.dirname(DIR_SPHINX_ROOT)
DIR_REPO_PARENT = os.path.dirname(DIR_REPO_ROOT)
DIR_PACKAGE_SRC = os.path.join(DIR_REPO_ROOT, "src")
# get all subdirectories of DIR_PACKAGE_SRC
PACKAGE_NAMES = [
    package
    for package in os.listdir(DIR_PACKAGE_SRC)
    if (
        not package.startswith("_")
        and os.path.isdir(os.path.join(DIR_PACKAGE_SRC, package))
    )
]
assert len(PACKAGE_NAMES) == 1, "only one package per Sphinx build is supported"
PROJECT_NAME = PACKAGE_NAMES[0]
EXCLUDE_MODULES = ["api"]
DIR_DOCS = os.path.join(DIR_REPO_ROOT, "docs")
DIR_DOCS_VERSION = os.path.join(DIR_DOCS, "docs-version")
DIR_SPHINX_SOURCE = os.path.join(DIR_SPHINX_ROOT, "source")
DIR_SPHINX_AUX = os.path.join(DIR_SPHINX_ROOT, "auxiliary")
DIR_SPHINX_API_GENERATED = os.path.join(DIR_SPHINX_SOURCE, "apidoc")
DIR_SPHINX_GENERATED = os.path.join(DIR_SPHINX_SOURCE, "_generated")
DIR_SPHINX_BUILD = os.path.join(DIR_SPHINX_ROOT, "build")
DIR_SPHINX_BUILD_DOCS_VERSION = os.path.join(DIR_SPHINX_BUILD, "docs-version")
DIR_SPHINX_BUILD_HTML = os.path.join(DIR_SPHINX_BUILD, "html")
DIR_SPHINX_TEMPLATES = os.path.join(DIR_SPHINX_ROOT, "make", "_templates")
DIR_SPHINX_TUTORIAL = os.path.join(DIR_SPHINX_SOURCE, "tutorial")
DIR_PACKAGE_ROOT = os.path.abspath(os.path.join(DIR_REPO_ROOT, "src", PROJECT_NAME))

FILE_AUTOSUMMARY_TEMPLATE = os.path.join(DIR_SPHINX_GENERATED, "autosummary.rst_")
FILE_JS_VERSIONS_RELATIVE = os.path.join("_static", "js", "versions.js")
FILE_JS_VERSIONS = os.path.join(DIR_SPHINX_BASE, FILE_JS_VERSIONS_RELATIVE)

# Environment variables
# noinspection SpellCheckingInspection
ENV_PYTHON_PATH = "PYTHONPATH"
ENV_SPHINX_BUILD = "SPHINX_BUILD"

# Version of the package being built
PACKAGE_VERSION = get_package_version(package_path=DIR_PACKAGE_ROOT)


class CommandMeta(ABCMeta):
    """
    Metaclass for command classes.

    Subsequent instantiations of a command class return the identical object.
    """

    def __init__(cls, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        cls.__instance_ref: ReferenceType[Command] | None = None

    def __call__(cls, *args: Any, **kwargs: Any) -> Command:
        """
        Return the existing command instance, or create a new one if none exists yet.

        :return: the command instance
        """
        if args or kwargs:
            raise ValueError("command classes may not take any arguments")

        if cls.__instance_ref:
            obj = cls.__instance_ref()
            if obj is not None:
                return obj

        instance: Command = super().__call__()
        cls.__instance_ref = ReferenceType(instance)
        return instance


class Command(metaclass=CommandMeta):
    """Defines an available command that can be launched from this module."""

    __RE_CAMEL_TO_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")

    def __init__(self) -> None:
        self.name = self.__RE_CAMEL_TO_SNAKE.sub("_", type(self).__name__).lower()

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_dependencies(self) -> tuple[Command, ...]:
        pass

    def get_prerequisites(self) -> Iterable[Command]:
        dependencies_extended: list[Command] = []

        for dependency in self.get_dependencies():
            dependencies_inherited = dependency.get_dependencies()
            if self in dependencies_inherited:
                raise ValueError(
                    f"circular dependency: {dependency.name} depends on {self.name}"
                )
            dependencies_extended.extend(
                dependency
                for dependency in dependencies_inherited
                if dependency not in dependencies_extended
            )
            dependencies_extended.append(dependency)

        return dependencies_extended

    def run(self) -> None:
        command_message = f"Running command {self.name} – {self.get_description()}"
        separator = "=" * len(command_message)
        log("\n".join(["", separator, command_message, separator]))
        self._run()

    @abstractmethod
    def _run(self) -> None:
        pass


#
# commands
#


class Clean(Command):
    def get_description(self) -> str:
        return "remove Sphinx build output"

    def get_dependencies(self) -> tuple[Command, ...]:
        return ()

    def _run(self) -> None:
        if os.path.exists(DIR_SPHINX_BUILD):
            shutil.rmtree(path=DIR_SPHINX_BUILD)
        if os.path.exists(DIR_SPHINX_API_GENERATED):
            shutil.rmtree(path=DIR_SPHINX_API_GENERATED)
        if os.path.exists(DIR_SPHINX_GENERATED):
            shutil.rmtree(path=DIR_SPHINX_GENERATED)


class ApiDoc(Command):
    def get_description(self) -> str:
        return "generate Sphinx API documentation from sources"

    def get_dependencies(self) -> tuple[Command, ...]:
        # noinspection PyRedundantParentheses
        return (Clean(),)

    def _run(self) -> None:
        packages = [
            package for package in os.listdir(DIR_PACKAGE_SRC) if package[:1].isalnum()
        ]
        log(f"Generating api documentation for {', '.join(packages)}")

        package_lines = "\n   ".join(packages)
        # noinspection SpellCheckingInspection
        autosummary_rst = f""".. autosummary::
   :toctree: ../apidoc
   :template: custom-module-template.rst
   :recursive:

   {package_lines}
"""  # noqa E241

        os.makedirs(os.path.dirname(FILE_AUTOSUMMARY_TEMPLATE), exist_ok=True)
        with open(FILE_AUTOSUMMARY_TEMPLATE, "w") as f:
            f.writelines(autosummary_rst)

        autogen_options = " ".join(
            [
                "-t",  # template path
                quote_path(DIR_SPHINX_TEMPLATES),
                "-i",  # include imports
                # the autosummary source file
                quote_path(FILE_AUTOSUMMARY_TEMPLATE),
            ]
        )

        subprocess.run(
            args=f"{CMD_SPHINX_AUTOGEN} {autogen_options}",
            shell=True,
            check=True,
        )

        # Adjust the path and filename as per your project's structure
        api_doc_filename = os.path.join(DIR_SPHINX_API_GENERATED, f"{packages[0]}.rst")
        new_title = "API Reference"
        title_underline = "=" * len(new_title)

        # Read the current content of the API doc file
        with open(api_doc_filename, "r+") as f:
            content = f.read()

            # Prepare the new header with the _api tag and the API Reference title
            new_header = f".. _api:\n\n{new_title}\n{title_underline}\n\n"

            # Use regex to remove the original package title and its underline
            # Assuming the title is the first thing after any potential whitespace
            # and followed by an underline of "=" characters
            content = re.sub(
                r"^\s*.*\n=+\n\n", "", content, count=1, flags=re.MULTILINE
            )

            # Combine the new header with the modified content
            new_content = new_header + content

            # Write the modified content back to the file
            f.seek(0)
            f.write(new_content)
            f.truncate()


class HomeDoc(Command):
    def get_description(self) -> str:
        return "generate home page from sources"

    def get_dependencies(self) -> tuple[Command, ...]:
        # noinspection PyRedundantParentheses
        return (Clean(),)

    def _run(self) -> None:
        # make dir if it does not exist
        os.makedirs(DIR_SPHINX_GENERATED, exist_ok=True)

        # open the rst readme file
        with open(os.path.join(DIR_REPO_ROOT, "README.rst")) as file:
            readme_data = file.read()

        # modify links (step back needed as build will add subdirectory to paths)
        readme_data = readme_data.replace("sphinx/source/", "/")

        # Remove the image block
        readme_data = re.sub(
            r"\.\. image::.*?\n\s*\n", "", readme_data, count=1, flags=re.S
        )

        # Remove badges
        readme_data = re.sub(
            r"\.\. Begin-Badges.*?\.\. End-Badges", "", readme_data, flags=re.S
        )

        # Replace the first title with "Home" and an appropriate underline
        readme_data = re.sub(
            r"^.+?\n=+\n", "Home\n====\n", readme_data, count=1, flags=re.S
        )

        with open(os.path.join(DIR_SPHINX_GENERATED, "release_notes.rst"), "w") as dst:
            dst.write(".. _release-notes:\n\n")
            with open(os.path.join(DIR_REPO_ROOT, "RELEASE_NOTES.rst")) as src:
                dst.write(src.read())

        with open(
            os.path.join(DIR_SPHINX_GENERATED, "home.rst"),
            "w",
        ) as file:
            file.write(".. _home:\n\n")
            file.writelines(readme_data)


class FetchPkgVersions(Command):
    def get_description(self) -> str:
        return "fetch available package versions with docs"

    def get_dependencies(self) -> tuple[Command, ...]:
        return ()

    def _run(self) -> None:
        versions = get_versions()

        latest_version = versions.latest_version
        version_data = {
            "current": str(latest_version) if latest_version else None,
            "all": list(map(str, versions.version_tags)),
        }

        version_data_as_js = (
            f"const DOCS_VERSIONS = {json.dumps(version_data, indent=4,)}"
        )

        with open(FILE_JS_VERSIONS, "w") as f:
            f.write(version_data_as_js)

        log(f"Version data written to: {FILE_JS_VERSIONS!r}")


class PrepareDocsDeployment(Command):
    def get_description(self) -> str:
        return "integrate documentation of previous versions"

    def get_dependencies(self) -> tuple[Command, ...]:
        return ()

    def _run(self) -> None:
        # we expect to find all previous documentation in docs/
        # and the newly built documentation in sphinx/build/

        # copy the new docs version to the deployment path
        self._copy_new_documentation()

        self._update_historic_docs()

        self._copy_latest_to_docs_root()

        self._tidy_up_docs_root()

    @staticmethod
    def _copy_new_documentation() -> None:
        dir_docs_current_version = os.path.join(
            DIR_DOCS_VERSION,
            version_string_to_url(PACKAGE_VERSION),
        )

        log(
            f"Copying new documentation from {DIR_SPHINX_BUILD_HTML!r} "
            f"to documentation version history at {dir_docs_current_version!r}"
        )

        os.makedirs(DIR_DOCS_VERSION, exist_ok=True)

        if os.path.exists(dir_docs_current_version):
            # remove a previous version of the same documentation if it exists
            shutil.rmtree(dir_docs_current_version)

        # copy new docs version to deployment path
        shutil.copytree(src=DIR_SPHINX_BUILD_HTML, dst=dir_docs_current_version)

    @staticmethod
    def _update_historic_docs() -> None:
        # Replace all docs version lists with the most up-to-date to have all versions
        # accessible also from older versions
        new_versions_js = os.path.join(
            DIR_DOCS_VERSION,
            version_string_to_url(PACKAGE_VERSION),
            FILE_JS_VERSIONS_RELATIVE,
        )

        for d in glob(os.path.join(DIR_DOCS_VERSION, "*")):
            old_versions_js = os.path.join(d, FILE_JS_VERSIONS_RELATIVE)
            if old_versions_js != new_versions_js:
                log(
                    "Copying 'versions.js' file from "
                    f"{new_versions_js!r} to {old_versions_js!r}"
                )
                os.makedirs(os.path.dirname(old_versions_js), exist_ok=True)
                shutil.copyfile(src=new_versions_js, dst=old_versions_js)

    @staticmethod
    def _copy_latest_to_docs_root() -> None:
        # copy the latest version to root

        latest_version = get_versions().latest_version
        if latest_version is None:
            raise RuntimeError("No documentation in place")

        dir_latest_version = os.path.join(
            DIR_DOCS_VERSION, version_string_to_url(latest_version)
        )

        log(
            f"Copying latest documentation from {dir_latest_version!r} "
            f"to documentation root at {DIR_DOCS!r}"
        )

        shutil.copytree(src=dir_latest_version, dst=DIR_DOCS, dirs_exist_ok=True)

    @staticmethod
    def _tidy_up_docs_root() -> None:
        # remove .buildinfo which interferes with GitHub Pages build
        file_build_info = os.path.join(DIR_DOCS, ".buildinfo")
        if os.path.exists(file_build_info):
            log(f"Removing file {file_build_info!r}")
            os.remove(file_build_info)

        # create empty file to signal that no GitHub auto-rendering is required
        # noinspection SpellCheckingInspection
        file_no_jekyll = os.path.join(DIR_DOCS, ".nojekyll")
        log(f"Creating empty file {file_no_jekyll!r}")
        open(file_no_jekyll, "a").close()


class Html(Command):
    def get_description(self) -> str:
        return "build Sphinx docs as HTML"

    def get_dependencies(self) -> tuple[Command, ...]:
        return Clean(), FetchPkgVersions(), ApiDoc(), HomeDoc()

    def _run(self) -> None:
        check_sphinx_version()

        os.makedirs(DIR_SPHINX_BUILD, exist_ok=True)

        sphinx_html_opts = [
            "-M html",
            quote_path(DIR_SPHINX_SOURCE),
            quote_path(DIR_SPHINX_BUILD),
        ]

        subprocess.run(
            args=f"{CMD_SPHINX_BUILD} {' '.join(sphinx_html_opts)}",
            shell=True,
            check=True,
        )


class Help(Command):
    def get_description(self) -> str:
        return "print this help message"

    def get_dependencies(self) -> tuple[Command, ...]:
        return ()

    def _run(self) -> None:
        print_usage()


class Versions:
    """
    Helper class that lists all versions that have already been released.
    """

    def __init__(self, version_tags: Iterable[pkg_version.Version]) -> None:
        self.version_tags = sorted(version_tags, reverse=True)

    @property
    def latest_version(self) -> pkg_version.Version | None:
        """
        Get the latest published version.

        :return:
        """
        return self.version_tags[0] if len(self.version_tags) > 0 else None


_versions: Versions | None = None


def get_versions() -> Versions:
    global _versions

    if _versions is not None:
        return _versions

    os.makedirs(DIR_SPHINX_BUILD, exist_ok=True)
    start_from_version_tag: pkg_version.Version = pkg_version.Version("0.0")

    version_tags: Iterable[pkg_version.Version] = (
        pkg_version.parse(s)
        for s in (
            subprocess.run(
                args='git tag -l "*.*.*"',
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode("UTF-8")
            .split("\n")
        )
        if s
    )

    # append the version we are building to version_tags
    version_tags = (*version_tags, PACKAGE_VERSION)

    versions_by_minor_version: dict[str, list[pkg_version.Version]] = defaultdict(list)

    for v in version_tags:
        if not (v.is_prerelease or v.is_devrelease) and v >= start_from_version_tag:
            versions_by_minor_version[f"{v.major}.{v.minor}"].append(v)

    minor_versions: list[pkg_version.Version] = [
        max(versions) for versions in versions_by_minor_version.values()
    ]
    # condition: if minor versions are not found, use the current version
    if not minor_versions:
        minor_versions = [PACKAGE_VERSION]

    log(f"Found minor versions: {', '.join(map(str, minor_versions))}")

    return Versions(minor_versions)


def make() -> None:
    """
    Run this make script with the given arguments.
    """
    if len(sys.argv) < 2:
        print_usage()

    commands_passed = sys.argv[1:]

    unknown_commands = set(commands_passed) - available_commands.keys()

    if unknown_commands:
        log(f"Unknown build commands: {' '.join(unknown_commands)}\n")
        print_usage()
        exit(1)

    # set up the Python path
    module_paths = [os.path.abspath(os.path.join(DIR_REPO_ROOT, "src"))]
    if ENV_PYTHON_PATH in os.environ:
        module_paths.append(os.environ[ENV_PYTHON_PATH])
    os.environ[ENV_PYTHON_PATH] = os.pathsep.join(module_paths)
    os.environ[ENV_SPHINX_BUILD] = "True"

    # run all given commands:
    executed_commands: set[Command] = set()

    for next_command_name in commands_passed:
        next_command: Command = available_commands[next_command_name]

        for prerequisite_command in next_command.get_prerequisites():
            if prerequisite_command not in executed_commands:
                prerequisite_command.run()
                executed_commands.add(prerequisite_command)

        next_command.run()
        executed_commands.add(next_command)


def log(message: Any) -> None:
    """
    Write a message to `stderr`.

    :param message: the message to write
    """
    print(str(message), file=sys.stderr)


def quote_path(path: str) -> str:
    """
    Quote a file path if it contains whitespace.
    """
    if " " in path or "\t" in path:
        return f'"{path}"'
    else:
        return path


def version_string_to_url(version: pkg_version.Version) -> str:
    """
    Make a Python package version string safe for URLs/folders.

    Our convention is to only replace all dots with dashes.
    """
    return f"{version.major}-{version.minor}"


def check_sphinx_version() -> None:
    import sphinx

    sphinx_version = pkg_version.parse(
        sphinx.__version__  # type:ignore[attr-defined]
    )
    if sphinx_version < pkg_version.parse("4.5"):
        raise RuntimeError("please upgrade sphinx to version 4.5 or newer")


def print_usage() -> None:
    usage = """Sphinx documentation build script
=================================

Available program arguments:
"""
    usage += "\n".join(
        f"\t{name} – {command.get_description()}"
        for name, command in available_commands.items()
    )
    print(usage)


available_commands: dict[str, Command] = {
    cmd.name: cmd
    for cmd in (
        Clean(),
        ApiDoc(),
        HomeDoc(),
        Html(),
        Help(),
        FetchPkgVersions(),
        PrepareDocsDeployment(),
    )
}
