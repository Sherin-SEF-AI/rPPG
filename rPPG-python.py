import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import time
from scipy import signal
from scipy.fft import fft, fftfreq
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings('ignore')

class RPPGApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Remote Photoplethysmography (rPPG) Heart Rate Monitor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize variables
        self.is_running = False
        self.camera = None
        self.camera_index = 0
        self.available_cameras = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Signal processing variables
        self.signal_buffer = []
        self.timestamps = []
        self.fps = 30
        self.buffer_size = 150  # 5 seconds at 30fps
        self.heart_rate = 0
        self.status = "Ready"
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Setup GUI
        self.setup_gui()
        
        # Detect available cameras
        self.detect_cameras()
        
    def setup_gui(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="Remote Photoplethysmography (rPPG) Heart Rate Monitor",
            font=('Arial', 24, 'bold'),
            fg='#ff6b6b',
            bg='#1a1a1a'
        )
        title_label.pack(pady=(0, 20))
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#1a1a1a')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video
        left_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video label
        self.video_label = tk.Label(left_panel, bg='#2d2d2d')
        self.video_label.pack(padx=10, pady=10)
        
        # Create placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (45, 45, 45)
        cv2.putText(placeholder, "Camera feed will appear here", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        self.update_video_display(placeholder)
        
        # Right panel - Controls
        right_panel = tk.Frame(content_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Camera selection
        camera_frame = tk.Frame(right_panel, bg='#2d2d2d')
        camera_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(camera_frame, text="Select Camera:", font=('Arial', 12), 
                fg='white', bg='#2d2d2d').pack(anchor=tk.W)
        
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(
            camera_frame, 
            textvariable=self.camera_var,
            font=('Arial', 11),
            state='readonly',
            width=30
        )
        self.camera_dropdown.pack(fill=tk.X, pady=(5, 0))
        self.camera_dropdown.bind('<<ComboboxSelected>>', self.on_camera_change)
        
        # Style for ttk widgets
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TCombobox', fieldbackground='#3d3d3d', background='#3d3d3d')
        
        # Start/Stop button
        self.start_button = tk.Button(
            right_panel,
            text="Start Monitoring",
            command=self.toggle_monitoring,
            font=('Arial', 14, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            relief=tk.RAISED,
            bd=0,
            padx=20,
            pady=15,
            cursor='hand2'
        )
        self.start_button.pack(fill=tk.X, padx=20, pady=20)
        
        # Status display
        status_frame = tk.Frame(right_panel, bg='#3d3d3d', relief=tk.SUNKEN, bd=2)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(status_frame, text="Status", font=('Arial', 12, 'bold'),
                fg='#ff6b6b', bg='#3d3d3d').pack(pady=(10, 5))
        
        self.status_label = tk.Label(
            status_frame,
            text=self.status,
            font=('Arial', 14),
            fg='white',
            bg='#3d3d3d'
        )
        self.status_label.pack(pady=(0, 10))
        
        # Heart rate display
        self.hr_frame = tk.Frame(right_panel, bg='#ff6b6b', relief=tk.RAISED, bd=2)
        self.hr_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(self.hr_frame, text="❤", font=('Arial', 30),
                fg='white', bg='#ff6b6b').pack(pady=(10, 0))
        
        self.hr_label = tk.Label(
            self.hr_frame,
            text="-- BPM",
            font=('Arial', 28, 'bold'),
            fg='white',
            bg='#ff6b6b'
        )
        self.hr_label.pack()
        
        tk.Label(self.hr_frame, text="Beats Per Minute", font=('Arial', 10),
                fg='white', bg='#ff6b6b').pack(pady=(0, 10))
        
        # Signal plot
        self.figure = Figure(figsize=(5, 2), dpi=80)
        self.figure.patch.set_facecolor('#2d2d2d')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_title('PPG Signal', color='white', fontsize=10)
        self.ax.set_xlabel('Time (s)', color='white', fontsize=8)
        self.ax.set_ylabel('Intensity', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=8)
        
        self.canvas = FigureCanvasTkAgg(self.figure, right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Instructions
        instructions_frame = tk.Frame(right_panel, bg='#3d3d3d', relief=tk.SUNKEN, bd=2)
        instructions_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(instructions_frame, text="Instructions", font=('Arial', 12, 'bold'),
                fg='#ff6b6b', bg='#3d3d3d').pack(pady=(10, 5))
        
        instructions = [
            "• Ensure good, consistent lighting",
            "• Position your face clearly in view",
            "• Minimize head and body movement",
            "• Wait 5-10 seconds for accurate reading",
            "• Green box shows detected face"
        ]
        
        for instruction in instructions:
            tk.Label(instructions_frame, text=instruction, font=('Arial', 10),
                    fg='white', bg='#3d3d3d', anchor='w').pack(fill=tk.X, padx=10, pady=2)
        
        tk.Label(instructions_frame, text="", bg='#3d3d3d').pack(pady=5)
        
        # Disclaimer
        disclaimer_frame = tk.Frame(main_frame, bg='#4a4a00', relief=tk.RAISED, bd=2)
        disclaimer_frame.pack(fill=tk.X, pady=(20, 0))
        
        disclaimer_text = (
            "DISCLAIMER: This application is for informational and demonstration purposes only. "
            "It is NOT a medical device and should not be used for diagnosis, treatment, or any medical purposes. "
            "Always consult with healthcare professionals for medical advice."
        )
        
        tk.Label(
            disclaimer_frame,
            text=disclaimer_text,
            font=('Arial', 10),
            fg='#ffff99',
            bg='#4a4a00',
            wraplength=1150,
            justify=tk.LEFT
        ).pack(padx=10, pady=10)
    
    def detect_cameras(self):
        """Detect available cameras"""
        self.available_cameras = []
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self.available_cameras.append(i)
                cap.release()
        
        if self.available_cameras:
            camera_names = [f"Camera {i} (Index: {i})" for i in self.available_cameras]
            self.camera_dropdown['values'] = camera_names
            self.camera_dropdown.current(0)
            self.camera_index = self.available_cameras[0]
        else:
            messagebox.showerror("Error", "No cameras detected!")
    
    def on_camera_change(self, event):
        """Handle camera selection change"""
        selected_index = self.camera_dropdown.current()
        if selected_index >= 0:
            self.camera_index = self.available_cameras[selected_index]
            if self.is_running:
                self.stop_monitoring()
                time.sleep(0.5)
                self.start_monitoring()
    
    def toggle_monitoring(self):
        """Toggle monitoring on/off"""
        if self.is_running:
            self.stop_monitoring()
        else:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start the monitoring process"""
        self.is_running = True
        self.signal_buffer = []
        self.timestamps = []
        self.heart_rate = 0
        self.status = "Initializing camera..."
        self.update_status()
        
        # Update button
        self.start_button.config(text="Stop Monitoring", bg='#f44336', activebackground='#da190b')
        
        # Start camera thread
        self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.camera_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        # Start GUI update
        self.update_gui()
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_running = False
        self.status = "Ready"
        self.update_status()
        
        # Update button
        self.start_button.config(text="Start Monitoring", bg='#4CAF50', activebackground='#45a049')
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # Clear plot
        self.ax.clear()
        self.ax.set_facecolor('#1a1a1a')
        self.ax.set_title('PPG Signal', color='white', fontsize=10)
        self.ax.set_xlabel('Time (s)', color='white', fontsize=8)
        self.ax.set_ylabel('Intensity', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=8)
        self.canvas.draw()
    
    def camera_worker(self):
        """Camera capture thread"""
        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Put frame in queue for processing
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, time.time()))
            else:
                self.status = "Camera error"
                break
            
            time.sleep(1/30)  # 30 FPS
    
    def processing_worker(self):
        """Process frames and calculate heart rate"""
        while self.is_running:
            try:
                # Get frame from queue
                frame, timestamp = self.frame_queue.get(timeout=1)
                
                # Convert to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect face
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Get the largest face
                    face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = face
                    
                    # Draw face rectangle
                    cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Define ROI (forehead region)
                    roi_x = x + int(w * 0.3)
                    roi_y = y + int(h * 0.05)
                    roi_w = int(w * 0.4)
                    roi_h = int(h * 0.2)
                    
                    # Draw ROI rectangle
                    cv2.rectangle(frame_rgb, (roi_x, roi_y), 
                                (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
                    
                    # Extract ROI
                    roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
                    
                    if roi.size > 0:
                        # Calculate mean green channel value
                        green_mean = np.mean(roi[:, :, 1])
                        
                        # Add to buffer
                        self.signal_buffer.append(green_mean)
                        self.timestamps.append(timestamp)
                        
                        # Keep buffer size limited
                        if len(self.signal_buffer) > self.buffer_size:
                            self.signal_buffer.pop(0)
                            self.timestamps.pop(0)
                        
                        # Process signal if we have enough samples
                        if len(self.signal_buffer) >= self.buffer_size:
                            self.process_signal()
                            self.status = f"Heart Rate: {self.heart_rate} BPM"
                        else:
                            progress = int((len(self.signal_buffer) / self.buffer_size) * 100)
                            self.status = f"Collecting data... {progress}%"
                else:
                    self.status = "No face detected"
                    cv2.putText(frame_rgb, "Position your face in view", (150, 240),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                # Put processed frame in result queue
                if not self.result_queue.full():
                    self.result_queue.put(frame_rgb)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def process_signal(self):
        """Process the PPG signal to extract heart rate"""
        if len(self.signal_buffer) < self.buffer_size:
            return
        
        # Convert to numpy array
        signal_array = np.array(self.signal_buffer)
        
        # Detrend signal
        detrended = signal.detrend(signal_array)
        
        # Calculate actual sampling rate
        if len(self.timestamps) > 1:
            time_diff = self.timestamps[-1] - self.timestamps[0]
            actual_fps = len(self.timestamps) / time_diff
        else:
            actual_fps = 30
        
        # Bandpass filter (0.8-3 Hz for 48-180 BPM)
        nyquist = actual_fps / 2
        low = 0.8 / nyquist
        high = 3.0 / nyquist
        
        if high < 1.0:  # Ensure high frequency is below Nyquist
            b, a = signal.butter(4, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, detrended)
        else:
            filtered = detrended
        
        # Normalize signal
        filtered = (filtered - np.mean(filtered)) / np.std(filtered)
        
        # Apply window to reduce spectral leakage
        window = signal.windows.hamming(len(filtered))
        windowed = filtered * window
        
        # Compute FFT
        fft_vals = np.abs(fft(windowed))
        freqs = fftfreq(len(windowed), 1/actual_fps)
        
        # Only consider positive frequencies in valid range
        valid_range = (freqs >= 0.8) & (freqs <= 3.0)
        valid_freqs = freqs[valid_range]
        valid_fft = fft_vals[valid_range]
        
        if len(valid_fft) > 0:
            # Find peak frequency
            peak_idx = np.argmax(valid_fft)
            peak_freq = valid_freqs[peak_idx]
            
            # Convert to BPM
            bpm = int(peak_freq * 60)
            
            # Validate BPM
            if 48 <= bpm <= 180:
                self.heart_rate = bpm
        
        # Update plot
        self.update_plot(filtered)
    
    def update_plot(self, signal_data):
        """Update the signal plot"""
        self.ax.clear()
        self.ax.set_facecolor('#1a1a1a')
        
        # Plot signal
        time_axis = np.linspace(0, len(signal_data) / 30, len(signal_data))
        self.ax.plot(time_axis, signal_data, 'g-', linewidth=1)
        
        self.ax.set_title('PPG Signal', color='white', fontsize=10)
        self.ax.set_xlabel('Time (s)', color='white', fontsize=8)
        self.ax.set_ylabel('Intensity', color='white', fontsize=8)
        self.ax.tick_params(colors='white', labelsize=8)
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def update_video_display(self, frame):
        """Update video display"""
        # Resize frame to fit label
        height, width = frame.shape[:2]
        max_width = 640
        max_height = 480
        
        # Calculate scaling factor
        scale = min(max_width/width, max_height/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize frame
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Convert to PIL format
        from PIL import Image, ImageTk
        image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update label
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def update_status(self):
        """Update status label"""
        self.status_label.config(text=self.status)
    
    def update_heart_rate_display(self):
        """Update heart rate display"""
        if self.heart_rate > 0:
            self.hr_label.config(text=f"{self.heart_rate} BPM")
        else:
            self.hr_label.config(text="-- BPM")
    
    def update_gui(self):
        """Update GUI elements"""
        if self.is_running:
            # Update video display
            try:
                frame = self.result_queue.get_nowait()
                self.update_video_display(frame)
            except queue.Empty:
                pass
            
            # Update displays
            self.update_status()
            self.update_heart_rate_display()
            
            # Schedule next update
            self.root.after(33, self.update_gui)  # ~30 FPS

def main():
    root = tk.Tk()
    app = RPPGApplication(root)
    root.mainloop()

if __name__ == "__main__":
    main()