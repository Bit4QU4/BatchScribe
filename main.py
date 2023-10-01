import time, os, ast, whisper, threading, queue, json, subprocess, sys
import torch
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar, Button, Label, Scale, Checkbutton
from ttkbootstrap import Style
torch.set_num_threads(1)
from tkinter.scrolledtext import ScrolledText
from concurrent.futures import ThreadPoolExecutor

class ToolTip:
    """
    ToolTip class for tkinter widgets.

    This class provides a simple way to show a tooltip when hovering over a tkinter widget.
    
    Usage:
        widget = tk.Label(root, text="Example")
        tooltip = ToolTip(widget, "This is a tooltip example")

    Parameters:
    - widget: The tkinter widget to which the tooltip is attached.
    - text: The text displayed inside the tooltip.
    """
    def __init__(self, widget, text):
        # Initialize the ToolTip with the associated widget and text to display.
        self.widget = widget
        self.text = text
        self.tooltip = None
        
        # Bind mouse entering and leaving events to the widget.
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        # Calculate the position for displaying the tooltip based on the widget's position.
        x, y, _, _ = self.widget.bbox("insert")  # Get widget bounding box coordinates.
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Create a top-level window for the tooltip.
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)  # Remove window decorations (like title bar).
        self.tooltip.wm_geometry(f"+{x}+{y}")  # Set window position.

        # Add a label to the tooltip window and display the text.
        label = tk.Label(self.tooltip, text=self.text, background="white", relief="solid", borderwidth=1, font=("Arial", "10", "normal"))
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        # Destroy the tooltip window if it exists.
        if self.tooltip:
            self.tooltip.destroy()

class WhisperTranscriberApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Whisper Transcriber')
        
        # Create a ttkbootstrap style object
        style = Style(theme='darkly')
        style.configure('TButton', padding=5)
        
        self.file_paths = tk.StringVar()
        self.file_label = ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.file_label.grid(row=0, column=0, padx=10, pady=10, sticky="we")

        button_frame = tk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.select_button = Button(button_frame, text='Select Files', command=self.select_files)
        self.select_button.grid(row=0, column=0, padx=1, pady=1)

        self.transcribe_button = Button(button_frame, text='Transcribe Files', command=self.transcribe_files)
        self.transcribe_button.grid(row=0, column=1, padx=1, pady=1)

        self.stop_button = Button(button_frame, text='Stop', command=self.stop_transcription)
        self.stop_button.grid(row=0, column=2, padx=2, pady=2)
        self.clear_button = Button(button_frame, text='Clear List', command=self.clear_list)
        self.clear_button.grid(row=0, column=3, padx=1, pady=1)
        

        self.root.columnconfigure(0, weight=1)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)

        self.worker_label = Label(self.root, text="Max Workers:")
        self.worker_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        ToolTip(self.worker_label, "Number of threads to run, > count requires better card")

        self.max_workers = tk.IntVar(value=5)
        # Label to display the current value of the slider
        self.worker_value_label = ttk.Label(self.root, text=self.max_workers.get())
        self.worker_value_label.grid(row=3, column=1, sticky="w")
        self.worker_slider = Scale(self.root, from_=2, to=10, orient=tk.HORIZONTAL, variable=self.max_workers, length=200)
        self.worker_slider.grid(row=3, column=0, sticky="we", padx=10, pady=5)
        self.max_workers.trace("w", self.update_worker_value_label)

        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=4, column=0, sticky="we", padx=10, pady=10)

        # Setup device that will be used for calc
        self.device = "cuda"
        if not self.check_nvidia_gpu():
            messagebox.showwarning("GPU Not Detected", "A compatible NVIDIA GPU was not detected. CUDA operations might not be supported. Using CPU fall-back. (CPU will likely be slower than GPU!)")
            # Use CPU as fallback if GPU is not available.
            self.device = "cpu"  
        self.results_queue = queue.Queue()
        self.job_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers.get())

        self.output_formats = {
            "vtt": tk.BooleanVar(),
            "srt": tk.BooleanVar(),
            "txt": tk.BooleanVar()
        }

        if not self.check_ffmpeg_installed():
            messagebox.showwarning("Missing Dependency", "ffmpeg is not installed on this system. Please install it to use this application.")
            sys.exit(1)

        row_num = 5  # Start from the next row after progress bar
        for format, var in self.output_formats.items():
            cb = tk.Checkbutton(self.root, text=f"Output {format.upper()}", variable=var)
            cb.grid(row=row_num, column=0, sticky="w", padx=10, pady=5)
            row_num += 1
    def update_worker_value_label(self, *args):
        # Existing code to update the label
        self.worker_value_label.config(text=self.max_workers.get())
        
        # Re-initialize the executor with the updated max_workers value
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers.get())

    def check_nvidia_gpu(self):
        # Check if the system has compatable GPU
        if not torch.cuda.is_available():
            return False
        device_name = torch.cuda.get_device_name(0)
        if "NVIDIA" in device_name:
            return True
        return False

    def check_ffmpeg_installed(self):
        # Check if ffmpeg is available as it's required for whisper
        try:
            result = subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                return True
            else:
                return False
        except FileNotFoundError:
            return False
    def clear_list(self):
        self.file_label.delete(1.0, tk.END)  # clear all text from the ScrolledText widget
        self.file_paths = ()  # also clear the stored file paths

    def select_files(self):
        self.clear_list()
        self.file_paths = filedialog.askopenfilenames(filetypes=(("All files", "*.*"), ("MP4 Files", "*.mp4"), ("MP3 Files", "*.mp3")))

        if len(self.file_paths) > 10:  # If there are more than 10 files, only display the first and last 5
            display_paths = self.file_paths[:5] + ("...",) + self.file_paths[-5:]
        else:  # If there are 10 or fewer files, display them all
            display_paths = self.file_paths

        self.file_label.insert(tk.INSERT, f'Selected file paths: {display_paths}\n')

    def transcribe_files(self):
        # Start workers
        self.start_time = time.time()
        for file_path in self.file_paths:
            self.job_queue.put(file_path)

        self.progress['maximum'] = len(self.file_paths)
        threading.Thread(target=self.worker_thread).start()
        self.root.after(1000, self.check_results_queue)  # Check the results queue every second

    def stop_transcription(self):
        # Ties back to the stop button
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers.get())

    def worker_thread(self):
        self.controls_to_lock = [self.select_button, self.transcribe_button, self.clear_button, self.worker_slider]
        
        for widget in self.controls_to_lock:
            widget.config(state=tk.DISABLED)

        while not self.job_queue.empty():
            future = self.executor.submit(self.transcribe_file, self.job_queue.get())
            future.add_done_callback(lambda x: self.root.after(0, self.update_progress))

        self.executor.shutdown(wait=True)  # Wait for all threads to finish
        self.end_time = time.time()  # Record the end time
        elapsed_time = self.end_time - self.start_time  # Calculate elapsed time
        
        # Re-enable the controls after all the work is done.
        self.root.after(0, self.re_enable_controls)
        # Convert elapsed time to hours, minutes, and seconds
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Build a formatted string based on the elapsed time
        if hours:
            formatted_time = f"{int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds"
        elif minutes:
            formatted_time = f"{int(minutes)} minutes and {seconds:.2f} seconds"
        else:
            formatted_time = f"{seconds:.2f} seconds"
        
        messagebox.showinfo("Done", f"Done transcribing. It took {formatted_time}.")
    def re_enable_controls(self):
        for widget in self.controls_to_lock:
            widget.config(state=tk.NORMAL)
    def transcribe_file(self, file_path):
        print(f"Transcribing file: {file_path}")
        try:
            # Load the model in the thread
            model = whisper.load_model("small").to(self.device)
            transcription_response = model.transcribe(file_path, language='en', verbose=True)
            for format, var in self.output_formats.items():
                if var.get():
                    output_filename = os.path.splitext(file_path)[0] + "." + format
                    if format == "txt":
                        with open(output_filename, 'w') as f:
                            # Extracting the 'text' field along with 'start' and 'end' timestamps
                            text_content = transcription_response.get('text', 'No transcription available')
                            start_time = transcription_response.get('start', 'N/A')
                            end_time = transcription_response.get('end', 'N/A')
                            formatted_content = f"**Start Time:** {start_time}\n{text_content}"
                            f.write(formatted_content)
                    else:
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

if __name__ == '__main__':
    app = WhisperTranscriberApp()
    app.run()