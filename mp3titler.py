import os
import eyed3
import re

def sanitize_filename(filename):
    # Remove illegal characters for Windows filenames
    return re.sub(r'[\\/:"*?<>|]', '', filename)

def rename_files_in_dir(dir_path):
    # List all files in directory
    for filename in os.listdir(dir_path):
        # Check if file is an mp3
        if filename.endswith('.mp3'):
            filepath = os.path.join(dir_path, filename)

            # Load the mp3 file in eyed3
            audiofile = eyed3.load(filepath)

            # Check if title tag is not None
            if audiofile.tag.title is not None:
                # Create new filename from title
                sanitized_title = sanitize_filename(audiofile.tag.title)
                new_filename = f"{sanitized_title}.mp3"
                new_filepath = os.path.join(dir_path, new_filename)

                # Rename file
                os.rename(filepath, new_filepath)

directory_path = ".\\download"
rename_files_in_dir(directory_path)
