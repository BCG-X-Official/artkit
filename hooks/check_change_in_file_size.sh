#!/bin/sh
echo "Checking change in file size for each staged file..."

MAX_FILE_DIFF_SIZE=1048576  # 1MB

# Get a list of all files with their statuses
git diff --cached --name-status | while IFS=$'\t' read status file optional_new_file; do
    case "$status" in
    D)  # If the file is deleted, skip the check
        echo "Skipping size check for deleted file $file."
        continue
        ;;
    R*) # Handle renamed files
        old_file="$file"
        new_file="$optional_new_file"
        echo "File renamed from $old_file to $new_file"
        file="$new_file"
        ;;
    esac

    echo "Processing $file with status $status"

    # Calculate the size of changes
    diff_size=$(git diff --cached -- "$file" | wc -c)
    echo "Diff size for $file: $diff_size bytes"

    # Check if the diff size for the file exceeds the limit
    if [ "$diff_size" -gt "$MAX_FILE_DIFF_SIZE" ]; then
        echo "Error: Change in file size for $file exceeds the limit of 1 MB."
        exit 1
    fi
done
