DEBUG = True

import cProfile  # Add this import statement
import time, os, ast, threading, queue, json, sys
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import filedialog, messagebox
from tkinter.ttk import Progressbar, Button, Label, Scale, Checkbutton
from ttkbootstrap import Style
# Sets the number of threads to use, by default torch tries to multithread; this bypasses it
from tkinter.scrolledtext import ScrolledText
import concurrent.futures  # Add this import statement
from concurrent.futures import ThreadPoolExecutor  # Add this import statement

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
        self.root.title('AV Voice2Text')
        self.root.resizable(False,False)
        style = Style(theme='yeti')
        style.configure('TButton', padding=5)
        # Initialize a StringVar to hold the file paths
        self.file_paths = tk.StringVar()
        
        # Create a ScrolledText widget to display the selected file paths
        self.file_label = ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.file_label.grid(row=0, column=0, padx=10, pady=10, sticky="we")

        # Create a frame to hold the buttons
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Create and place the 'Select Files' button
        self.select_button = Button(button_frame, text='Select Files', command=self.select_files)
        self.select_button.grid(row=0, column=0, padx=1, pady=1)

        # Create and place the 'Transcribe Files' button
        self.transcribe_button = Button(button_frame, text='Transcribe Files', command=self.transcribe_files)
        self.transcribe_button.grid(row=0, column=1, padx=1, pady=1)

        # Create and place the 'Stop' button
        self.stop_button = Button(button_frame, text='Stop', command=self.stop_transcription)
        self.stop_button.grid(row=0, column=2, padx=2, pady=2)

        # Create and place the 'Clear List' button
        self.clear_button = Button(button_frame, text='Clear List', command=self.clear_list)
        self.clear_button.grid(row=0, column=3, padx=1, pady=1)

        # Configure column weights to ensure proper resizing
        self.root.columnconfigure(0, weight=1)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)

        self.worker_label = Label(self.root, text="Max Workers:")
        self.worker_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

        ToolTip(self.worker_label, "Number of threads to run, > count requires better graphics card")
        
        self.max_workers = tk.IntVar(value=5)
        # Label to display the current value of the slider
        self.worker_value_label = ttk.Label(self.root, text=self.max_workers.get())
        self.worker_value_label.grid(row=3, column=1, sticky="w")
        self.worker_slider = Scale(self.root, from_=2, to=10, orient=tk.HORIZONTAL, variable=self.max_workers, length=200)
        self.worker_slider.grid(row=3, column=0, sticky="we", padx=10, pady=5)
        self.max_workers.trace("w", self.update_worker_value_label)

        self.progress = Progressbar(self.root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=4, column=0, sticky="we", padx=10, pady=10)
        ToolTip(self.progress, "Shows the current progress of conversion, in terms of # FILEs completed")

        # Create startup pop-up
        messagebox.showinfo("Startup", f"Hi, {os.environ.get('USERNAME') or os.environ.get('USER')}.\n To use this software start by selecting the files you want to use, then hit 'Start Transcription'. Please note that the progress bar is per file completion.")

        # Setup device that will be used for calc
        if not self.check_nvidia_gpu():
            messagebox.showwarning("GPU Not Detected", "A compatible NVIDIA GPU was not detected. CUDA operations might not be supported. Using CPU fall-back. (CPU will likely be slower than GPU!)")
            # Use CPU as fallback if GPU is not available.
            self.device = "cpu"
        else:
            self.device = "cuda"
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
        load_heavy_modules()
        if not torch.cuda.is_available():
            return False

        # Get the name and compute capability of the first GPU device
        device_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)

        # Check if the device is an NVIDIA GPU and meets the minimum CUDA capability
        min_compute_capability = (4, 0)
        if "NVIDIA" in device_name and compute_capability >= min_compute_capability:
            return True
        # If the device is not an NVIDIA GPU or doesn't meet the requirements
        return False
    
    def install_cuda_drivers(self):
        if not self.check_nvidia_gpu():
            messagebox.showwarning("GPU Not Detected", "A compatible NVIDIA GPU was not detected. CUDA operations might not be supported. Using CPU fall-back. (CPU will likely be slower than GPU!)")
            return False

        try:
            # Attempt to install CUDA drivers
            result = subprocess.run(["sudo", "apt-get", "install", "nvidia-cuda-toolkit"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if result.returncode == 0:
                messagebox.showinfo("CUDA Installation", "CUDA drivers installed successfully.")
                return True
            else:
                messagebox.showerror("CUDA Installation Failed", f"Failed to install CUDA drivers. Error: {result.stderr}")
                return False
        except Exception as e:
            messagebox.showerror("CUDA Installation Error", f"An error occurred while trying to install CUDA drivers: {str(e)}")
            return False
        
    def get_ffmpeg_path(self):
        if getattr(sys, 'frozen', False):
            # If the application is bundled with PyInstaller
            base_path = sys._MEIPASS
            print("Frozen! Checking FFMpeg location")
            print(str(base_path))
        else:
            # If running in a normal Python environment
            base_path = os.path.dirname(os.path.abspath(__file__))
            print("Detected DEV Env for FFMpeg")
            print(str(base_path))
        return os.path.join(base_path, 'ffmpeg.exe')

    def check_ffmpeg_installed(self):
        # Check if ffmpeg is available as it's required for whisper
        try:
            result = subprocess.run([self.get_ffmpeg_path(), "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
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
        self.file_paths = filedialog.askopenfilenames(
        filetypes=(
            ("Video Files", "*.mp4;*.avi;*.mov;*.flv;*.wmv"),  # Add or remove video formats as needed
            ("Audio Files", "*.mp3;*.wav;*.aac;*.flac;*.ogg"),  # Add or remove audio formats as needed
            ("All Files", "*.*")
        )
    )
        if len(self.file_paths) > 10:  # If there are more than 10 files, only display the first and last 5
            display_paths = [os.path.basename(path) for path in self.file_paths[:5]] + ["..."] + [os.path.basename(path) for path in self.file_paths[-5:]]
        elif len(self.file_paths) == 0:
            display_paths = "No files selected!"
        else:  # If there are 10 or fewer files, display them all
            display_paths = [os.path.basename(path) for path in self.file_paths]

        # Format display paths for insertion into label
        if isinstance(display_paths, list):
            display_paths = '\n '.join(display_paths)

        self.file_label.insert(tk.INSERT, f'Selected file paths: {display_paths}\n')

    def transcribe_files(self):
        if not any(var.get() for var in self.output_formats.values()):
            messagebox.showerror("No Format", message="No format selected, select an output format to continue.")
        else:
            if DEBUG ==True:
                pr = cProfile.Profile()
                pr.enable()
            self.start_time = time.time()
            for file_path in self.file_paths:
                self.job_queue.put(file_path)
            self.progress['maximum'] = len(self.file_paths)
            threading.Thread(target=self.worker_thread).start()
            # Check the results queue every second
            self.root.after(1000, self.check_results_queue)
            if DEBUG == True:
                pr.disable()
                pr.dump_stats(f'transcribe_file_{os.path.basename(file_path)}.prof')

    def stop_transcription(self):
        # Ties back to the stop button
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers.get())

    def worker_thread(self):
        if DEBUG ==True:
            pr = cProfile.Profile()
            pr.enable()
        self.controls_to_lock = [self.select_button, self.transcribe_button, self.clear_button, self.worker_slider]
        # Start control lockout to prevent user tamper during transcript 
        for widget in self.controls_to_lock:
            widget.config(state=tk.DISABLED)

        # Initialize the executor here
        with ThreadPoolExecutor(max_workers=self.max_workers.get()) as executor:
            while not self.job_queue.empty():
                future = executor.submit(self.transcribe_file, self.job_queue.get())
                future.add_done_callback(lambda x: self.root.after(0, self.update_progress))

            # The executor will automatically shut down here
            # wait=True is implicit with the context manager

        self.end_time = time.time()  # Record the end time
        elapsed_time = self.end_time - self.start_time  # Calculate elapsed time

        # Re-enable the controls after all the work is done
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

        if DEBUG == True:
            pr.disable()
            pr.dump_stats('worker_thread_profile.prof')
        messagebox.showinfo("Done", f"Done transcribing. It took {formatted_time}.")

    def re_enable_controls(self):
        for widget in self.controls_to_lock:
            widget.config(state=tk.NORMAL)

    def transcribe_file(self, file_path):
        load_heavy_modules()
        print(f"Transcribing file: {file_path}")
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load the model in the thread
            if getattr(sys, 'frozen', False):
                # If running in a PyInstaller bundle
                base_path = sys._MEIPASS
            else:
                # Normal Python environment
                base_path = os.path.dirname(os.path.abspath(__file__))

            model_path = os.path.join(base_path, 'models')
            try:
                model = whisper.load_model(name="small", download_root=model_path).to(self.device)
            except Exception as model_error:
                raise RuntimeError(f"Failed to load model: {model_error}")

            transcription_response = model.transcribe(file_path, language='en', verbose=True)

            for format, var in self.output_formats.items():
                if var.get():
                    output_filename = os.path.splitext(file_path)[0] + "." + format
                    output_directory = os.path.dirname(output_filename)
                    if not os.path.exists(output_directory):
                        os.makedirs(output_directory)

                    if format == "txt":
                        try:
                            with open(output_filename, 'w') as f:
                                text_content = transcription_response.get('text', 'No transcription available')
                                start_time = transcription_response.get('start', 'N/A')
                                end_time = transcription_response.get('end', 'N/A')
                                formatted_content = f"**Start Time:** {start_time}\n{text_content}"
                                f.write(formatted_content)
                        except IOError as io_error:
                            raise IOError(f"Failed to write to file {output_filename}: {io_error}")
                    else:
                        try:
                            output_writer = whisper.utils.get_writer(format, output_directory)
                            output_writer(transcription_response, output_filename)
                        except Exception as output_error:
                            raise Exception(f"Error in writing output for format {format}: {output_error}")

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

def load_heavy_modules():
    global whisper, torch, subprocess
    import whisper
    import torch
    import subprocess

if __name__ == '__main__':
    app = WhisperTranscriberApp()
    app.run()