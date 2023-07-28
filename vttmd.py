import os
import webvtt
from datetime import timedelta

def convert_time(time):
    # Convert time into MM:SS format
    return str(timedelta(seconds=time))

def convert_vtt_to_md(vtt_filename, md_filename):
    vtt = webvtt.read(vtt_filename)

    with open(md_filename, 'w', encoding='utf-8') as f:
        for caption in vtt:
            start_time = convert_time(caption.start_in_seconds)
            end_time = convert_time(caption.end_in_seconds)
            f.write(f"**{start_time} - {end_time}**\n\n")
            f.write(caption.text + "\n\n")

def convert_all(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.vtt'):
            base = os.path.splitext(filename)[0]
            convert_vtt_to_md(os.path.join(directory, filename), os.path.join(directory, base + '.md'))

# Example usage:
convert_all(input('path_to_your_directory:\n'))