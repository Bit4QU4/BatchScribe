import os
import re
import sys

import eyed3

# Names Windows refuses regardless of extension (CON.mp3 is still invalid)
_WINDOWS_RESERVED = (
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)


def sanitize_filename(filename):
    name = re.sub(r'[\\/:"*?<>|\x00-\x1f]', '', filename)
    # Windows strips trailing dots/spaces itself, which can cause collisions
    name = name.rstrip(' .')
    if not name:
        name = 'untitled'
    elif name.split('.')[0].upper() in _WINDOWS_RESERVED:
        name = f'_{name}'
    # Leave headroom for the extension and dedup suffix within the 255 limit
    return name[:240]

def rename_files_in_dir(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith('.mp3'):
            filepath = os.path.join(dir_path, filename)
            audiofile = eyed3.load(filepath)
            if audiofile is None or audiofile.tag is None or audiofile.tag.title is None:
                continue
            sanitized_title = sanitize_filename(audiofile.tag.title)
            new_filename = f"{sanitized_title}.mp3"
            new_filepath = os.path.join(dir_path, new_filename)
            counter = 2
            while os.path.exists(new_filepath) and new_filepath != filepath:
                new_filename = f"{sanitized_title}_{counter}.mp3"
                new_filepath = os.path.join(dir_path, new_filename)
                counter += 1
            if new_filepath != filepath:
                os.rename(filepath, new_filepath)

if __name__ == '__main__':
    rename_files_in_dir(sys.argv[1] if len(sys.argv) > 1 else input('Directory path: '))
