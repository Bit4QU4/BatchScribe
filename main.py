import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar
import os
import ast
import whisper
import threading
import queue

class WhisperTranscriberApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Whisper Transcriber')

        self.file_paths = tk.StringVar()
        self.file_label = tk.Label(self.root, text='No file selected')
        self.file_label.pack()

        self.select_button = tk.Button(self.root, text='Select Files', command=self.select_files)
        self.select_button.pack()

        self.transcribe_button = tk.Button(self.root, text='Transcribe Files', command=self.transcribe_files)
        self.transcribe_button.pack()

        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='indeterminate')
        self.progress.pack()

        self.model = None
        try:
            self.model = whisper.load_model("medium")
        except Exception as e:
            messagebox.showerror("Error", "Could not load whisper model: " + str(e))

        self.job_queue = queue.Queue()

        self.output_formats = {
            "vtt": tk.BooleanVar(),
            "srt": tk.BooleanVar(),
        }
        for format, var in self.output_formats.items():
            cb = tk.Checkbutton(self.root, text=f"Output {format.upper()}", variable=var)
            cb.pack()

    def select_files(self):
        file_paths = filedialog.askopenfilenames(filetypes=(("MP4 Files", "*.mp4"), ("MP3 Files", "*.mp3"), ("All files", "*.*")))
        self.file_paths.set(str(file_paths))  # Convert tuple to string
        self.file_label.config(text=f'Selected file paths: {file_paths}')

    def transcribe_files(self):
        file_paths = ast.literal_eval(self.file_paths.get())  # Convert string back to tuple

        for file_path in file_paths:
            self.job_queue.put(file_path)

        self.progress.start()
        threading.Thread(target=self.worker_thread).start()

    def worker_thread(self):
        while not self.job_queue.empty():
            file_path = self.job_queue.get()
            print(f"Transcribing file: {file_path}")
            try:
                transcription_response = self.model.transcribe(file_path, language='en', verbose=True)
                for format, var in self.output_formats.items():
                    if var.get():
                        output_filename = os.path.splitext(file_path)[0] + "." + format
                        output_writer = whisper.utils.get_writer(format, os.path.dirname(file_path))
                        output_writer(transcription_response, output_filename)
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred while transcribing: {str(e)}")

        self.progress.stop()
                
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = WhisperTranscriberApp()
    app.run()