#!/usr/bin/env python3
"""
Text-to-Speech Converter
A standalone script that allows selecting text files via a file dialog
and converting them to MP3 files using OpenAI's TTS API.
"""

import os
import re
import html
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import threading

import markdown
from openai import OpenAI

# ───────────────────────── Configuration ─────────────────────────
MAX_CHARS = 4_000  # TTS limit is 4_096 characters
VOICES = ["alloy", "nova", "shimmer", "fable", "echo", "onyx"]
MODELS = ["tts-1", "tts-1-hd"]

# ──────────────────────── Helper utilities ───────────────────────
def strip_markdown(md: str) -> str:
    """Very lightweight Markdown → plain‑text conversion."""
    html_text = markdown.markdown(md)
    plain = re.sub(r"<[^>]+>", " ", html_text)
    return html.unescape(plain)


def read_file(file_path: Path) -> str:
    """Read and process text file content."""
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()
    return strip_markdown(raw) if file_path.suffix.lower() == '.md' else raw


def chunk_text(text: str, size: int = MAX_CHARS):
    """Split text into chunks of maximum size."""
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n and text[end] not in {" ", "\n"}:
            end = text.rfind(" ", start, end) or end
        yield text[start:end].strip()
        start = end


def synthesize(text: str, model: str, voice: str, outfile: Path, progress_callback=None):
    """Stream each ≤4 k‑char chunk to *outfile* as MP3."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Count chunks for progress reporting
    chunks = list(chunk_text(text))
    total_chunks = len(chunks)
    
    for i, block in enumerate(chunks):
        file_mode = "wb" if i == 0 else "ab"
        with client.audio.speech.with_streaming_response.create(
            model=model,
            voice=voice,
            input=block,
            response_format="mp3",
        ) as resp, open(outfile, file_mode) as f:
            for chunk in resp.iter_bytes():
                f.write(chunk)
                
        # Update progress
        if progress_callback:
            progress = int(((i + 1) / total_chunks) * 100)
            progress_callback(progress)


# ─────────────────────── GUI Application ──────────────────────────
class TTSConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Text to Speech Converter")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.selected_files = []  # Store multiple selected files
        self.setup_ui()
        
    def setup_ui(self):
        # Create a main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # API Key
        api_frame = ttk.Frame(main_frame)
        api_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(api_frame, text="OpenAI API Key:").pack(side=tk.LEFT)
        self.api_key_var = tk.StringVar(value=self.api_key)
        self.api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, show="*", width=40)
        self.api_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Files selection section
        files_frame = ttk.LabelFrame(main_frame, text="Files Selection", padding=10)
        files_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Files list with scrollbar
        files_list_frame = ttk.Frame(files_frame)
        files_list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(files_list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.files_listbox = tk.Listbox(files_list_frame, selectmode=tk.EXTENDED, height=7)
        self.files_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.files_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.files_listbox.yview)
        
        # Buttons for file management
        files_button_frame = ttk.Frame(files_frame)
        files_button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(files_button_frame, text="Add Files", command=self.browse_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(files_button_frame, text="Remove Selected", command=self.remove_selected_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(files_button_frame, text="Clear All", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        
        # Options Frame
        options_frame = ttk.LabelFrame(main_frame, text="TTS Options", padding=10)
        options_frame.pack(fill=tk.X, pady=10)
        
        # Voice Selection
        voice_frame = ttk.Frame(options_frame)
        voice_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(voice_frame, text="Voice:").pack(side=tk.LEFT)
        self.voice_var = tk.StringVar(value=VOICES[0])
        ttk.Combobox(voice_frame, textvariable=self.voice_var, values=VOICES, state="readonly").pack(
            side=tk.LEFT, padx=5
        )
        
        # Model Selection
        model_frame = ttk.Frame(options_frame)
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_var = tk.StringVar(value=MODELS[0])
        ttk.Combobox(model_frame, textvariable=self.model_var, values=MODELS, state="readonly").pack(
            side=tk.LEFT, padx=5
        )
        
        # Output directory
        output_frame = ttk.Frame(main_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        self.output_dir_var = tk.StringVar(value=str(Path.home()))
        ttk.Label(output_frame, text="Output Directory:").pack(side=tk.LEFT)
        ttk.Entry(output_frame, textvariable=self.output_dir_var, state="readonly", width=40).pack(
            side=tk.LEFT, padx=5, fill=tk.X, expand=True
        )
        ttk.Button(output_frame, text="Browse", command=self.browse_output_dir).pack(side=tk.RIGHT)
        
        # Progress section
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=10)
        
        # Current file label
        self.current_file_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.current_file_var).pack(fill=tk.X)
        
        # File progress
        ttk.Label(progress_frame, text="Current File:").pack(anchor=tk.W)
        self.file_progress_var = tk.IntVar(value=0)
        self.file_progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', 
            variable=self.file_progress_var
        )
        self.file_progress_bar.pack(fill=tk.X)
        
        # Overall progress
        ttk.Label(progress_frame, text="Overall Progress:").pack(anchor=tk.W, pady=(10, 0))
        self.overall_progress_var = tk.IntVar(value=0)
        self.overall_progress_bar = ttk.Progressbar(
            progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate', 
            variable=self.overall_progress_var
        )
        self.overall_progress_bar.pack(fill=tk.X)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
        # Convert button
        convert_button = ttk.Button(main_frame, text="Convert to MP3", command=self.convert_to_speech)
        convert_button.pack(pady=10)
    
    def browse_files(self):
        """Allow selection of multiple files."""
        filetypes = [("Text files", "*.txt"), ("Markdown files", "*.md"), ("All files", "*.*")]
        file_paths = filedialog.askopenfilenames(
            title="Select text files",
            filetypes=filetypes
        )
        if file_paths:
            for path in file_paths:
                if path not in self.selected_files:
                    self.selected_files.append(path)
                    self.files_listbox.insert(tk.END, Path(path).name)
    
    def remove_selected_files(self):
        """Remove selected files from the list."""
        selected_indices = self.files_listbox.curselection()
        if not selected_indices:
            return
            
        # Remove in reverse order to avoid index issues
        for i in sorted(selected_indices, reverse=True):
            del self.selected_files[i]
            self.files_listbox.delete(i)
    
    def clear_files(self):
        """Clear all files from the list."""
        self.selected_files.clear()
        self.files_listbox.delete(0, tk.END)
    
    def browse_output_dir(self):
        directory = filedialog.askdirectory(
            title="Select output directory"
        )
        if directory:
            self.output_dir_var.set(directory)
    
    def update_file_progress(self, value):
        """Update progress for current file."""
        self.file_progress_var.set(value)
        self.root.update_idletasks()  # Update the UI
    
    def update_overall_progress(self, value):
        """Update overall progress."""
        self.overall_progress_var.set(value)
        self.root.update_idletasks()  # Update the UI
    
    def convert_to_speech(self):
        if not self.selected_files:
            messagebox.showerror("Error", "Please select at least one text file")
            return
        
        # Set API key if provided
        api_key = self.api_key_var.get()
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        elif not os.getenv("OPENAI_API_KEY"):
            messagebox.showerror("Error", "OpenAI API key is required")
            return
        
        # Get options
        model = self.model_var.get()
        voice = self.voice_var.get()
        output_dir = self.output_dir_var.get()
        
        # Reset progress
        self.file_progress_var.set(0)
        self.overall_progress_var.set(0)
        self.status_var.set("Converting files...")
        
        # Run conversion in a separate thread to keep UI responsive
        def conversion_thread():
            total_files = len(self.selected_files)
            successful_conversions = 0
            failed_conversions = 0
            
            for i, file_path in enumerate(self.selected_files):
                input_file = Path(file_path)
                output_file = Path(output_dir) / f"{input_file.stem}.mp3"
                
                # Update status
                self.current_file_var.set(f"Processing: {input_file.name} ({i+1}/{total_files})")
                self.file_progress_var.set(0)
                
                try:
                    # Read and process the file
                    text = read_file(input_file)
                    
                    # Synthesize speech
                    synthesize(text, model, voice, output_file, self.update_file_progress)
                    
                    successful_conversions += 1
                    
                except Exception as e:
                    failed_conversions += 1
                    error_msg = f"Error processing {input_file.name}: {str(e)}"
                    self.status_var.set(error_msg)
                    messagebox.showerror("Conversion Error", error_msg)
                
                # Update overall progress
                overall_progress = int(((i + 1) / total_files) * 100)
                self.update_overall_progress(overall_progress)
            
            # Final status update
            if failed_conversions == 0:
                self.status_var.set(f"Done! Successfully converted {successful_conversions} files.")
                messagebox.showinfo("Success", f"All {successful_conversions} files were converted successfully!")
            else:
                self.status_var.set(f"Completed with {successful_conversions} successes and {failed_conversions} failures.")
                messagebox.showinfo("Partial Success", 
                                   f"Converted {successful_conversions} files successfully.\n"
                                   f"Failed to convert {failed_conversions} files.")
        
        threading.Thread(target=conversion_thread, daemon=True).start()


def main():
    root = tk.Tk()
    app = TTSConverterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()