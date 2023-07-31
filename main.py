import time
import torch
torch.set_num_threads(1)

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter.ttk import Progressbar
import os
import ast
import whisper
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

class WhisperTranscriberApp:

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Whisper Transcriber')

        self.file_paths = tk.StringVar()
        self.file_label = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.file_label.pack()

        self.select_button = tk.Button(self.root, text='Select Files', command=self.select_files)
        self.select_button.pack()

        self.transcribe_button = tk.Button(self.root, text='Transcribe Files', command=self.transcribe_files)
        self.transcribe_button.pack()

        self.stop_button = tk.Button(self.root, text='Stop', command=self.stop_transcription)
        self.stop_button.pack()

        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack()

        self.device = "cuda"
        self.results_queue = queue.Queue()
        self.job_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=8)

        self.output_formats = {
            "vtt": tk.BooleanVar(),
            "srt": tk.BooleanVar(),
        }
        for format, var in self.output_formats.items():
            cb = tk.Checkbutton(self.root, text=f"Output {format.upper()}", variable=var)
            cb.pack()

    def select_files(self):
        self.file_paths = filedialog.askopenfilenames(filetypes=(("MP4 Files", "*.mp4"), ("MP3 Files", "*.mp3"), ("All files", "*.*")))

        if len(self.file_paths) > 10:  # If there are more than 10 files, only display the first and last 5
            display_paths = self.file_paths[:5] + ("...",) + self.file_paths[-5:]
        else:  # If there are 10 or fewer files, display them all
            display_paths = self.file_paths

        self.file_label.insert(tk.INSERT, f'Selected file paths: {display_paths}\n')

    def transcribe_files(self):
        for file_path in self.file_paths:
            self.job_queue.put(file_path)

        self.progress['maximum'] = len(self.file_paths)
        threading.Thread(target=self.worker_thread).start()
        self.root.after(1000, self.check_results_queue)  # Check the results queue every second

    def stop_transcription(self):
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=8)

    def worker_thread(self):
        while not self.job_queue.empty():
            future = self.executor.submit(self.transcribe_file, self.job_queue.get())
            future.add_done_callback(lambda x: self.root.after(0, self.update_progress))

        self.executor.shutdown(wait=True)  # Wait for all threads to finish
        self.end_time = time.time()  # Record the end time
        elapsed_time = self.end_time - self.start_time  # Calculate elapsed time
        messagebox.showinfo("Done", f"Done transcribing. It took {elapsed_time} seconds.")  # Show a message box


    def transcribe_file(self, file_path):
        print(f"Transcribing file: {file_path}")
        try:
            # Load the model in the thread
            model = whisper.load_model("small").to(self.device)
            transcription_response = model.transcribe(file_path, language='en', verbose=True)
            for format, var in self.output_formats.items():
                if var.get():
                    output_filename = os.path.splitext(file_path)[0] + "." + format
                    output_writer = whisper.utils.get_writer(format, os.path.dirname(file_path))
                    output_writer(transcription_response, output_filename)
            self.results_queue.put(("success", f"Successfully transcribed file: {file_path}"))
        except Exception as e:
            self.results_queue.put(("error", f"An error occurred while transcribing: {str(e)}"))

    def check_results_queue(self):
        while not self.results_queue.empty():
            result_type, result = self.results_queue.get()
            if result_type == "error":
                messagebox.showinfo("Transcription Error", result)
        self.root.after(1000, self.check_results_queue)  # Check the results queue again in one second

    def update_progress(self):
        self.progress.step()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = WhisperTranscriberApp()
    app.run()