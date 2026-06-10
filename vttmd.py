import os
from datetime import timedelta

import webvtt


def convert_time(seconds):
    return str(timedelta(seconds=seconds))

def convert_vtt_to_md(vtt_filename, md_filename):
    vtt = webvtt.read(vtt_filename)
    with open(md_filename, 'w', encoding='utf-8') as f:
        for caption in vtt:
            start_time = convert_time(caption.start_in_seconds)
            end_time = convert_time(caption.end_in_seconds)
            f.write(f"**{start_time} - {end_time}**:")
            f.write(caption.text + "\n")

def convert_all(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.vtt'):
            base = os.path.splitext(filename)[0]
            md_path = os.path.join(directory, base + '.md')
            convert_vtt_to_md(os.path.join(directory, filename), md_path)

if __name__ == '__main__':
    convert_all(input('path_to_your_directory:\n'))
