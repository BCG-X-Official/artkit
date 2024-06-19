#!/usr/bin/env python3
import os

# Define the copyright notice
COPYRIGHT_NOTICE = """\
# -----------------------------------------------------------------------------
# © 2024 Boston Consulting Group. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -----------------------------------------------------------------------------
"""


def add_copyright_notice(file_path):
    with open(file_path) as file:
        content = file.read()

    if (
        COPYRIGHT_NOTICE.strip() not in content
    ):  # Avoid adding the notice if it's already present
        with open(file_path, "w") as file:
            file.write(COPYRIGHT_NOTICE + "\n" + content)


def recursively_add_notice_to_py_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                add_copyright_notice(file_path)


# Specify the directory you want to start the search from
start_directory = "../src"  # Replace with your directory path

recursively_add_notice_to_py_files(start_directory)

print("Copyright notice added to all .py files.")
