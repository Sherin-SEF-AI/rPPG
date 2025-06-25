import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import time
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import queue
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation
from datetime import datetime
import json
import os
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Advanced signal processing algorithms
class SignalProcessingAlgorithms:
    @staticmethod
    def green_channel(roi):
        """Traditional green channel method"""
        return np.mean(roi[:, :, 1])
    
    @staticmethod
    def chrom_method(roi):
        """CHROM-based rPPG method"""
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
        if np.any(mean_rgb == 0):
            return 0
        
        # Normalize
        norm_rgb = mean_rgb / np.sum(mean_rgb)
        
        # CHROM combination
        xs = 3 * norm_rgb[0] - 2 * norm_rgb[1]
        ys = 1.5 * norm_rgb[0] + norm_rgb[1] - 1.5 * norm_rgb[2]
        
        # Return combined signal
        return xs - ys
    
    @staticmethod
    def pos_method(roi):
        """Plane-Orthogonal-to-Skin (POS) method"""
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0)
        if np.any(mean_rgb == 0):
            return 0
        
        # Normalize channels
        norm_rgb = mean_rgb / np.mean(mean_rgb)
        
        # POS combination
        s = norm_rgb[1] - norm_rgb[2]
        
        return s
    
    @staticmethod
    def ica_method(roi):
        """Independent Component Analysis based method"""
        # Simplified ICA-like approach
        rgb_signals = np.mean(roi.reshape(-1, 3), axis=0)
        
        # Whiten the signals
        if np.std(rgb_signals) > 0:
            whitened = (rgb_signals - np.mean(rgb_signals)) / np.std(rgb_signals)
            # Simple ICA-like combination
            return whitened[1] - 0.5 * whitened[0] - 0.5 * whitened[2]
        return 0

class AdvancedRPPGApplication:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced rPPG Heart Rate Monitor - Professional Edition")
        self.root.geometry("1400x900")
        self.root.configure(bg='#0f0f0f')
        
        # Initialize variables
        self.is_running = False
        self.is_recording = False
        self.camera = None
        self.camera_index = 0
        self.available_cameras = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Signal processing variables
        self.signal_buffers = {
            'green': deque(maxlen=300),
            'chrom': deque(maxlen=300),
            'pos': deque(maxlen=300),
            'ica': deque(maxlen=300)
        }
        self.timestamps = deque(maxlen=300)
        self.fps_buffer = deque(maxlen=30)
        self.current_fps = 30
        self.heart_rates = {'green': 0, 'chrom': 0, 'pos': 0, 'ica': 0}
        self.final_heart_rate = 0
        self.confidence_score = 0
        self.status = "Ready"
        
        # Recording variables
        self.recorded_data = []
        self.recording_start_time = None
        
        # HRV Analysis
        self.rr_intervals = deque(maxlen=60)  # Store R-R intervals
        self.hrv_metrics = {
            'SDNN': 0,  # Standard deviation of NN intervals
            'RMSSD': 0,  # Root mean square of successive differences
            'pNN50': 0   # Percentage of successive intervals differing by >50ms
        }
        
        # Settings
        self.algorithm = tk.StringVar(value='ensemble')
        self.roi_size = tk.IntVar(value=30)
        self.filter_order = tk.IntVar(value=4)
        self.window_size = tk.IntVar(value=150)
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=5)
        
        # Setup GUI
        self.setup_gui()
        
        # Detect available cameras
        self.detect_cameras()
        
    def setup_gui(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Main monitoring tab
        self.main_tab = tk.Frame(self.notebook, bg='#0f0f0f')
        self.notebook.add(self.main_tab, text='Monitor')
        
        # Analytics tab
        self.analytics_tab = tk.Frame(self.notebook, bg='#0f0f0f')
        self.notebook.add(self.analytics_tab, text='Analytics')
        
        # Settings tab
        self.settings_tab = tk.Frame(self.notebook, bg='#0f0f0f')
        self.notebook.add(self.settings_tab, text='Settings')
        
        # History tab
        self.history_tab = tk.Frame(self.notebook, bg='#0f0f0f')
        self.notebook.add(self.history_tab, text='History')
        
        # Setup each tab
        self.setup_main_tab()
        self.setup_analytics_tab()
        self.setup_settings_tab()
        self.setup_history_tab()
        
        # Disclaimer at bottom
        self.setup_disclaimer()
    
    def setup_main_tab(self):
        # Main container
        main_frame = tk.Frame(self.main_tab, bg='#0f0f0f')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top section
        top_frame = tk.Frame(main_frame, bg='#0f0f0f')
        top_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title with gradient effect
        title_label = tk.Label(
            top_frame, 
            text="Advanced Remote Photoplethysmography System",
            font=('Arial', 28, 'bold'),
            fg='#ff6b6b',
            bg='#0f0f0f'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            top_frame,
            text="Multi-Algorithm Heart Rate & HRV Analysis",
            font=('Arial', 14),
            fg='#888888',
            bg='#0f0f0f'
        )
        subtitle_label.pack()
        
        # Content frame
        content_frame = tk.Frame(main_frame, bg='#0f0f0f')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Video
        left_panel = tk.Frame(content_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video frame with controls
        video_container = tk.Frame(left_panel, bg='#1a1a1a')
        video_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video label
        self.video_label = tk.Label(video_container, bg='#2d2d2d')
        self.video_label.pack()
        
        # Create placeholder
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (45, 45, 45)
        cv2.putText(placeholder, "Camera feed will appear here", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        self.update_video_display(placeholder)
        
        # FPS and camera info
        info_frame = tk.Frame(left_panel, bg='#1a1a1a')
        info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.fps_label = tk.Label(
            info_frame,
            text="FPS: --",
            font=('Arial', 10),
            fg='#00ff00',
            bg='#1a1a1a'
        )
        self.fps_label.pack(side=tk.LEFT)
        
        self.resolution_label = tk.Label(
            info_frame,
            text="Resolution: 640x480",
            font=('Arial', 10),
            fg='#00ff00',
            bg='#1a1a1a'
        )
        self.resolution_label.pack(side=tk.RIGHT)
        
        # Right panel - Controls and displays
        right_panel = tk.Frame(content_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2, width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        right_panel.pack_propagate(False)
        
        # Camera selection
        camera_frame = tk.Frame(right_panel, bg='#1a1a1a')
        camera_frame.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(camera_frame, text="Camera Selection:", font=('Arial', 12, 'bold'), 
                fg='#ff6b6b', bg='#1a1a1a').pack(anchor=tk.W)
        
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(
            camera_frame, 
            textvariable=self.camera_var,
            font=('Arial', 11),
            state='readonly',
            width=35
        )
        self.camera_dropdown.pack(fill=tk.X, pady=(5, 0))
        self.camera_dropdown.bind('<<ComboboxSelected>>', self.on_camera_change)
        
        # Control buttons
        button_frame = tk.Frame(right_panel, bg='#1a1a1a')
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.start_button = tk.Button(
            button_frame,
            text="▶ Start Monitoring",
            command=self.toggle_monitoring,
            font=('Arial', 12, 'bold'),
            bg='#4CAF50',
            fg='white',
            activebackground='#45a049',
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2'
        )
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.record_button = tk.Button(
            button_frame,
            text="⏺ Record",
            command=self.toggle_recording,
            font=('Arial', 12, 'bold'),
            bg='#ff4444',
            fg='white',
            activebackground='#cc0000',
            relief=tk.FLAT,
            bd=0,
            padx=20,
            pady=12,
            cursor='hand2',
            state=tk.DISABLED
        )
        self.record_button.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Status display
        status_frame = tk.Frame(right_panel, bg='#2d2d2d', relief=tk.SUNKEN, bd=2)
        status_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(status_frame, text="System Status", font=('Arial', 12, 'bold'),
                fg='#ff6b6b', bg='#2d2d2d').pack(pady=(10, 5))
        
        self.status_label = tk.Label(
            status_frame,
            text=self.status,
            font=('Arial', 14),
            fg='white',
            bg='#2d2d2d'
        )
        self.status_label.pack(pady=(0, 10))
        
        # Heart rate display with confidence
        self.hr_frame = tk.Frame(right_panel, bg='#ff6b6b', relief=tk.RAISED, bd=2)
        self.hr_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Heart icon with animation
        self.heart_label = tk.Label(self.hr_frame, text="❤", font=('Arial', 40),
                                   fg='white', bg='#ff6b6b')
        self.heart_label.pack(pady=(10, 0))
        
        self.hr_label = tk.Label(
            self.hr_frame,
            text="-- BPM",
            font=('Arial', 32, 'bold'),
            fg='white',
            bg='#ff6b6b'
        )
        self.hr_label.pack()
        
        # Confidence meter
        confidence_frame = tk.Frame(self.hr_frame, bg='#ff6b6b')
        confidence_frame.pack(pady=(5, 10))
        
        tk.Label(confidence_frame, text="Confidence:", font=('Arial', 10),
                fg='white', bg='#ff6b6b').pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(
            confidence_frame,
            text="0%",
            font=('Arial', 10, 'bold'),
            fg='white',
            bg='#ff6b6b'
        )
        self.confidence_label.pack(side=tk.LEFT, padx=(5, 0))
        
        # Algorithm results
        algo_frame = tk.Frame(right_panel, bg='#2d2d2d', relief=tk.SUNKEN, bd=2)
        algo_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(algo_frame, text="Algorithm Results", font=('Arial', 12, 'bold'),
                fg='#ff6b6b', bg='#2d2d2d').pack(pady=(10, 5))
        
        self.algo_labels = {}
        for algo in ['green', 'chrom', 'pos', 'ica']:
            frame = tk.Frame(algo_frame, bg='#2d2d2d')
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(frame, text=f"{algo.upper()}:", font=('Arial', 10),
                    fg='#888888', bg='#2d2d2d', width=8, anchor='w').pack(side=tk.LEFT)
            
            self.algo_labels[algo] = tk.Label(
                frame,
                text="-- BPM",
                font=('Arial', 10, 'bold'),
                fg='white',
                bg='#2d2d2d'
            )
            self.algo_labels[algo].pack(side=tk.LEFT)
        
        tk.Label(algo_frame, text="", bg='#2d2d2d').pack(pady=5)
        
        # HRV Metrics
        hrv_frame = tk.Frame(right_panel, bg='#2d2d2d', relief=tk.SUNKEN, bd=2)
        hrv_frame.pack(fill=tk.X, padx=20)
        
        tk.Label(hrv_frame, text="HRV Metrics", font=('Arial', 12, 'bold'),
                fg='#ff6b6b', bg='#2d2d2d').pack(pady=(10, 5))
        
        self.hrv_labels = {}
        for metric in ['SDNN', 'RMSSD', 'pNN50']:
            frame = tk.Frame(hrv_frame, bg='#2d2d2d')
            frame.pack(fill=tk.X, padx=10, pady=2)
            
            tk.Label(frame, text=f"{metric}:", font=('Arial', 10),
                    fg='#888888', bg='#2d2d2d', width=8, anchor='w').pack(side=tk.LEFT)
            
            self.hrv_labels[metric] = tk.Label(
                frame,
                text="--",
                font=('Arial', 10, 'bold'),
                fg='white',
                bg='#2d2d2d'
            )
            self.hrv_labels[metric].pack(side=tk.LEFT)
        
        tk.Label(hrv_frame, text="", bg='#2d2d2d').pack(pady=5)
    
    def setup_analytics_tab(self):
        # Create plots frame
        plots_frame = tk.Frame(self.analytics_tab, bg='#0f0f0f')
        plots_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Signal plot
        signal_frame = tk.Frame(plots_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        signal_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tk.Label(signal_frame, text="PPG Signal Analysis", font=('Arial', 14, 'bold'),
                fg='#ff6b6b', bg='#1a1a1a').pack(pady=10)
        
        self.signal_figure = Figure(figsize=(10, 3), dpi=80)
        self.signal_figure.patch.set_facecolor('#1a1a1a')
        self.signal_ax = self.signal_figure.add_subplot(111)
        self.signal_ax.set_facecolor('#0f0f0f')
        
        self.signal_canvas = FigureCanvasTkAgg(self.signal_figure, signal_frame)
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Spectrum plot
        spectrum_frame = tk.Frame(plots_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        spectrum_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(spectrum_frame, text="Frequency Spectrum", font=('Arial', 14, 'bold'),
                fg='#ff6b6b', bg='#1a1a1a').pack(pady=10)
        
        self.spectrum_figure = Figure(figsize=(10, 3), dpi=80)
        self.spectrum_figure.patch.set_facecolor('#1a1a1a')
        self.spectrum_ax = self.spectrum_figure.add_subplot(111)
        self.spectrum_ax.set_facecolor('#0f0f0f')
        
        self.spectrum_canvas = FigureCanvasTkAgg(self.spectrum_figure, spectrum_frame)
        self.spectrum_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        # Statistics panel
        stats_frame = tk.Frame(plots_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(stats_frame, text="Session Statistics", font=('Arial', 14, 'bold'),
                fg='#ff6b6b', bg='#1a1a1a').pack(pady=10)
        
        stats_container = tk.Frame(stats_frame, bg='#1a1a1a')
        stats_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        self.stats_labels = {}
        stats = ['Average HR', 'Min HR', 'Max HR', 'HR Variability', 'Signal Quality']
        
        for i, stat in enumerate(stats):
            frame = tk.Frame(stats_container, bg='#2d2d2d', relief=tk.RAISED, bd=1)
            frame.grid(row=0, column=i, padx=5, sticky='ew')
            stats_container.columnconfigure(i, weight=1)
            
            tk.Label(frame, text=stat, font=('Arial', 10),
                    fg='#888888', bg='#2d2d2d').pack(pady=(10, 5))
            
            self.stats_labels[stat] = tk.Label(
                frame,
                text="--",
                font=('Arial', 16, 'bold'),
                fg='#00ff00',
                bg='#2d2d2d'
            )
            self.stats_labels[stat].pack(pady=(0, 10))
    
    def setup_settings_tab(self):
        # Settings container
        settings_frame = tk.Frame(self.settings_tab, bg='#0f0f0f')
        settings_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=40)
        
        # Algorithm selection
        algo_frame = tk.LabelFrame(settings_frame, text="Algorithm Selection", 
                                  font=('Arial', 12, 'bold'), fg='#ff6b6b', 
                                  bg='#1a1a1a', relief=tk.RAISED, bd=2)
        algo_frame.pack(fill=tk.X, pady=(0, 20))
        
        algorithms = [
            ('Ensemble (All algorithms)', 'ensemble'),
            ('Green Channel', 'green'),
            ('CHROM Method', 'chrom'),
            ('POS Method', 'pos'),
            ('ICA Method', 'ica')
        ]
        
        for text, value in algorithms:
            tk.Radiobutton(
                algo_frame,
                text=text,
                variable=self.algorithm,
                value=value,
                font=('Arial', 11),
                fg='white',
                bg='#1a1a1a',
                selectcolor='#2d2d2d',
                activebackground='#1a1a1a',
                activeforeground='white'
            ).pack(anchor=tk.W, padx=20, pady=5)
        
        # Processing parameters
        param_frame = tk.LabelFrame(settings_frame, text="Processing Parameters", 
                                   font=('Arial', 12, 'bold'), fg='#ff6b6b', 
                                   bg='#1a1a1a', relief=tk.RAISED, bd=2)
        param_frame.pack(fill=tk.X, pady=(0, 20))
        
        # ROI Size
        roi_container = tk.Frame(param_frame, bg='#1a1a1a')
        roi_container.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(roi_container, text="ROI Size (%):", font=('Arial', 11),
                fg='white', bg='#1a1a1a').pack(side=tk.LEFT)
        
        roi_scale = tk.Scale(
            roi_container,
            from_=10, to=50,
            variable=self.roi_size,
            orient=tk.HORIZONTAL,
            font=('Arial', 10),
            fg='white',
            bg='#1a1a1a',
            highlightthickness=0,
            troughcolor='#2d2d2d'
        )
        roi_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Filter Order
        filter_container = tk.Frame(param_frame, bg='#1a1a1a')
        filter_container.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(filter_container, text="Filter Order:", font=('Arial', 11),
                fg='white', bg='#1a1a1a').pack(side=tk.LEFT)
        
        filter_scale = tk.Scale(
            filter_container,
            from_=2, to=8,
            variable=self.filter_order,
            orient=tk.HORIZONTAL,
            font=('Arial', 10),
            fg='white',
            bg='#1a1a1a',
            highlightthickness=0,
            troughcolor='#2d2d2d'
        )
        filter_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Window Size
        window_container = tk.Frame(param_frame, bg='#1a1a1a')
        window_container.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(window_container, text="Window Size:", font=('Arial', 11),
                fg='white', bg='#1a1a1a').pack(side=tk.LEFT)
        
        window_scale = tk.Scale(
            window_container,
            from_=90, to=300,
            variable=self.window_size,
            orient=tk.HORIZONTAL,
            font=('Arial', 10),
            fg='white',
            bg='#1a1a1a',
            highlightthickness=0,
            troughcolor='#2d2d2d'
        )
        window_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Export settings
        export_frame = tk.LabelFrame(settings_frame, text="Export Options", 
                                    font=('Arial', 12, 'bold'), fg='#ff6b6b', 
                                    bg='#1a1a1a', relief=tk.RAISED, bd=2)
        export_frame.pack(fill=tk.X)
        
        button_container = tk.Frame(export_frame, bg='#1a1a1a')
        button_container.pack(pady=20)
        
        tk.Button(
            button_container,
            text="Export Session Data",
            command=self.export_data,
            font=('Arial', 11),
            bg='#4CAF50',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            button_container,
            text="Generate Report",
            command=self.generate_report,
            font=('Arial', 11),
            bg='#2196F3',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=10)
    
    def setup_history_tab(self):
        # History container
        history_frame = tk.Frame(self.history_tab, bg='#0f0f0f')
        history_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Controls
        control_frame = tk.Frame(history_frame, bg='#0f0f0f')
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Button(
            control_frame,
            text="Load Session",
            command=self.load_session,
            font=('Arial', 11),
            bg='#4CAF50',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            control_frame,
            text="Clear History",
            command=self.clear_history,
            font=('Arial', 11),
            bg='#f44336',
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=10,
            cursor='hand2'
        ).pack(side=tk.LEFT)
        
        # History list
        list_frame = tk.Frame(history_frame, bg='#1a1a1a', relief=tk.RAISED, bd=2)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview
        self.history_tree = ttk.Treeview(
            list_frame,
            columns=('Date', 'Duration', 'Avg HR', 'Algorithm'),
            show='tree headings',
            height=15
        )
        
        # Configure columns
        self.history_tree.heading('#0', text='Session')
        self.history_tree.heading('Date', text='Date')
        self.history_tree.heading('Duration', text='Duration')
        self.history_tree.heading('Avg HR', text='Avg HR')
        self.history_tree.heading('Algorithm', text='Algorithm')
        
        self.history_tree.column('#0', width=100)
        self.history_tree.column('Date', width=150)
        self.history_tree.column('Duration', width=100)
        self.history_tree.column('Avg HR', width=100)
        self.history_tree.column('Algorithm', width=100)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=scrollbar.set)
        
        self.history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Load existing history
        self.load_history()
    
    def setup_disclaimer(self):
        disclaimer_frame = tk.Frame(self.root, bg='#4a4a00', relief=tk.RAISED, bd=2)
        disclaimer_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        disclaimer_text = (
            "⚠️ MEDICAL DISCLAIMER: This application is for research and demonstration purposes only. "
            "It is NOT a medical device and should not be used for diagnosis, treatment, or any medical purposes. "
            "The measurements provided may not be accurate. Always consult healthcare professionals for medical advice."
        )
        
        tk.Label(
            disclaimer_frame,
            text=disclaimer_text,
            font=('Arial', 10),
            fg='#ffff99',
            bg='#4a4a00',
            wraplength=1350,
            justify=tk.LEFT
        ).pack(padx=10, pady=10)
    
    def detect_cameras(self):
        """Detect all available cameras"""
        self.available_cameras = []
        # Continue from detect_cameras method
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    
                    camera_info = {
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps
                    }
                    self.available_cameras.append(camera_info)
                cap.release()
        
        if self.available_cameras:
            camera_names = [f"{cam['name']} ({cam['resolution']} @ {cam['fps']}fps)" 
                            for cam in self.available_cameras]
            self.camera_dropdown['values'] = camera_names
            self.camera_dropdown.current(0)
            self.camera_index = self.available_cameras[0]['index']
        else:
            messagebox.showerror("Error", "No cameras detected!")
    
    def on_camera_change(self, event):
        """Handle camera selection change"""
        selected_index = self.camera_dropdown.current()
        if selected_index >= 0:
            self.camera_index = self.available_cameras[selected_index]['index']
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
    
    def toggle_recording(self):
        """Toggle recording on/off"""
        if not self.is_recording:
            self.is_recording = True
            self.recording_start_time = time.time()
            self.recorded_data = []
            self.record_button.config(text="⏹ Stop Recording", bg='#880000')
            self.status = "Recording..."
        else:
            self.is_recording = False
            self.record_button.config(text="⏺ Record", bg='#ff4444')
            self.save_recording()
    
    def start_monitoring(self):
        """Start the monitoring process"""
        self.is_running = True
        
        # Clear buffers
        for buffer in self.signal_buffers.values():
            buffer.clear()
        self.timestamps.clear()
        self.rr_intervals.clear()
        
        # Reset metrics
        self.heart_rates = {'green': 0, 'chrom': 0, 'pos': 0, 'ica': 0}
        self.final_heart_rate = 0
        self.confidence_score = 0
        self.status = "Initializing camera..."
        self.update_status()
        
        # Update buttons
        self.start_button.config(text="⏸ Stop Monitoring", bg='#f44336')
        self.record_button.config(state=tk.NORMAL)
        
        # Start threads
        self.camera_thread = threading.Thread(target=self.camera_worker, daemon=True)
        self.camera_thread.start()
        
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        # Start GUI updates
        self.update_gui()
        self.animate_heart()
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_running = False
        self.is_recording = False
        self.status = "Ready"
        self.update_status()
        
        # Update buttons
        self.start_button.config(text="▶ Start Monitoring", bg='#4CAF50')
        self.record_button.config(text="⏺ Record", bg='#ff4444', state=tk.DISABLED)
        
        # Release camera
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        # Save session if data available
        if self.final_heart_rate > 0:
            self.save_session()
    
    def camera_worker(self):
        """Camera capture thread with enhanced processing"""
        self.camera = cv2.VideoCapture(self.camera_index)
        
        # Set camera properties for optimal performance
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
        
        # Auto exposure off if possible
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        frame_time = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if ret:
                current_time = time.time()
                
                # Calculate actual FPS
                fps = 1 / (current_time - frame_time)
                frame_time = current_time
                self.fps_buffer.append(fps)
                
                # Put frame in queue
                if not self.frame_queue.full():
                    self.frame_queue.put((frame, current_time))
            else:
                self.status = "Camera error"
                break
            
            # Maintain target FPS
            time.sleep(max(0, 1/30 - (time.time() - current_time)))
    
    def processing_worker(self):
        """Advanced processing thread with multiple algorithms"""
        face_detector = cv2.dnn.readNetFromCaffe(
            cv2.data.haarcascades + '../deploy.prototxt',
            cv2.data.haarcascades + '../res10_300x300_ssd_iter_140000.caffemodel'
        ) if os.path.exists(cv2.data.haarcascades + '../deploy.prototxt') else None
        
        while self.is_running:
            try:
                # Get frame from queue
                frame, timestamp = self.frame_queue.get(timeout=1)
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect face (use DNN if available, else cascade)
                face_roi = None
                if face_detector:
                    face_roi = self.detect_face_dnn(frame, face_detector)
                else:
                    face_roi = self.detect_face_cascade(frame)
                
                if face_roi is not None:
                    # Apply all algorithms
                    signals = {
                        'green': SignalProcessingAlgorithms.green_channel(face_roi),
                        'chrom': SignalProcessingAlgorithms.chrom_method(face_roi),
                        'pos': SignalProcessingAlgorithms.pos_method(face_roi),
                        'ica': SignalProcessingAlgorithms.ica_method(face_roi)
                    }
                    
                    # Add to buffers
                    for algo, value in signals.items():
                        self.signal_buffers[algo].append(value)
                    self.timestamps.append(timestamp)
                    
                    # Process signals if enough data
                    window_size = self.window_size.get()
                    if len(self.timestamps) >= window_size:
                        self.process_all_signals()
                        
                        # Calculate ensemble result
                        if self.algorithm.get() == 'ensemble':
                            self.calculate_ensemble_hr()
                        else:
                            algo = self.algorithm.get()
                            self.final_heart_rate = self.heart_rates.get(algo, 0)
                            self.confidence_score = self.calculate_confidence()
                        
                        # Update HRV if we have stable HR
                        if self.final_heart_rate > 0:
                            self.update_hrv_metrics()
                        
                        self.status = f"Heart Rate: {self.final_heart_rate} BPM"
                    else:
                        progress = int((len(self.timestamps) / window_size) * 100)
                        self.status = f"Collecting data... {progress}%"
                    
                    # Record data if recording
                    if self.is_recording and self.final_heart_rate > 0:
                        self.recorded_data.append({
                            'timestamp': timestamp,
                            'heart_rate': self.final_heart_rate,
                            'algorithms': self.heart_rates.copy(),
                            'confidence': self.confidence_score,
                            'hrv': self.hrv_metrics.copy()
                        })
                else:
                    self.status = "No face detected - Please position your face in view"
                
                # Put processed frame in result queue
                if not self.result_queue.full():
                    self.result_queue.put(frame_rgb)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def detect_face_cascade(self, frame):
        """Detect face using Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Get largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            
            # Draw face rectangle on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calculate ROI with dynamic size
            roi_size_percent = self.roi_size.get() / 100
            roi_x = x + int(w * (0.5 - roi_size_percent/2))
            roi_y = y + int(h * 0.05)
            roi_w = int(w * roi_size_percent)
            roi_h = int(h * 0.2)
            
            # Ensure ROI is within bounds
            roi_x = max(0, roi_x)
            roi_y = max(0, roi_y)
            roi_w = min(roi_w, frame.shape[1] - roi_x)
            roi_h = min(roi_h, frame.shape[0] - roi_y)
            
            # Draw ROI rectangle
            cv2.rectangle(frame, (roi_x, roi_y), 
                        (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
            
            # Extract and return ROI
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            return roi if roi.size > 0 else None
        
        return None
    
    def detect_face_dnn(self, frame, detector):
        """Detect face using DNN for better accuracy"""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        
        detector.setInput(blob)
        detections = detector.forward()
        
        # Find best detection
        best_confidence = 0
        best_face = None
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5 and confidence > best_confidence:
                best_confidence = confidence
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                best_face = box.astype("int")
        
        if best_face is not None:
            x, y, x2, y2 = best_face
            return self.extract_roi_from_coords(frame, x, y, x2-x, y2-y)
        
        return None
    
    def extract_roi_from_coords(self, frame, x, y, w, h):
        """Extract ROI from face coordinates"""
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Calculate ROI
        roi_size_percent = self.roi_size.get() / 100
        roi_x = x + int(w * (0.5 - roi_size_percent/2))
        roi_y = y + int(h * 0.05)
        roi_w = int(w * roi_size_percent)
        roi_h = int(h * 0.2)
        
        # Draw ROI
        cv2.rectangle(frame, (roi_x, roi_y), 
                    (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)
        
        # Extract ROI
        roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        return roi if roi.size > 0 else None
    
    def process_all_signals(self):
        """Process signals from all algorithms"""
        if len(self.timestamps) < 2:
            return
        
        # Calculate actual sampling rate
        time_diff = self.timestamps[-1] - self.timestamps[0]
        actual_fps = len(self.timestamps) / time_diff if time_diff > 0 else 30
        self.current_fps = actual_fps
        
        # Process each algorithm
        for algo, buffer in self.signal_buffers.items():
            if len(buffer) >= self.window_size.get():
                hr = self.process_single_signal(list(buffer), actual_fps, algo)
                self.heart_rates[algo] = hr
    
    def process_single_signal(self, signal_data, fs, algorithm_name):
        """Advanced signal processing for a single algorithm"""
        try:
            # Convert to numpy array
            signal_array = np.array(signal_data)
            
            # Remove NaN and inf values
            signal_array = signal_array[np.isfinite(signal_array)]
            if len(signal_array) < self.window_size.get():
                return 0
            
            # Detrend using polynomial fit
            x = np.arange(len(signal_array))
            coeffs = np.polyfit(x, signal_array, 3)
            trend = np.polyval(coeffs, x)
            detrended = signal_array - trend
            
            # Normalize
            if np.std(detrended) > 0:
                normalized = (detrended - np.mean(detrended)) / np.std(detrended)
            else:
                return 0
            
            # Advanced filtering
            filter_order = self.filter_order.get()
            
            # Bandpass filter
            nyquist = fs / 2
            low_freq = 0.7  # 42 BPM
            high_freq = 3.5  # 210 BPM
            
            if nyquist > high_freq:
                # Design filter
                sos = signal.butter(filter_order, [low_freq/nyquist, high_freq/nyquist], 
                                    btype='band', output='sos')
                filtered = signal.sosfiltfilt(sos, normalized)
            else:
                filtered = normalized
            
            # Apply window
            window = signal.windows.tukey(len(filtered), alpha=0.1)
            windowed = filtered * window
            
            # Compute power spectral density using Welch's method
            nperseg = min(len(windowed), int(fs * 5))  # 5 second segments
            freqs, psd = signal.welch(windowed, fs=fs, nperseg=nperseg, 
                                        noverlap=nperseg//2, scaling='density')
            
            # Find peak in valid range
            valid_range = (freqs >= 0.7) & (freqs <= 3.5)
            valid_freqs = freqs[valid_range]
            valid_psd = psd[valid_range]
            
            if len(valid_psd) > 0:
                # Find peaks
                peaks, properties = signal.find_peaks(valid_psd, 
                                                    height=np.max(valid_psd)*0.3,
                                                    distance=int(0.2*fs))
                
                if len(peaks) > 0:
                    # Get highest peak
                    peak_idx = peaks[np.argmax(valid_psd[peaks])]
                    peak_freq = valid_freqs[peak_idx]
                    
                    # Refine frequency estimate using parabolic interpolation
                    if 0 < peak_idx < len(valid_psd) - 1:
                        y1 = valid_psd[peak_idx - 1]
                        y2 = valid_psd[peak_idx]
                        y3 = valid_psd[peak_idx + 1]
                        
                        a = (y3 - y1) / 2
                        b = y2 - (y1 + y3) / 2
                        
                        if b != 0:
                            x_offset = -a / (2 * b)
                            peak_freq = valid_freqs[peak_idx] + x_offset * (valid_freqs[1] - valid_freqs[0])
                    
                    # Convert to BPM
                    bpm = int(peak_freq * 60)
                    
                    # Validate
                    if 45 <= bpm <= 200:
                        return bpm
            
            return 0
            
        except Exception as e:
            print(f"Error processing {algorithm_name} signal: {e}")
            return 0
    
    def calculate_ensemble_hr(self):
        """Calculate ensemble heart rate using weighted average"""
        # Algorithm weights based on typical accuracy
        weights = {
            'green': 0.2,
            'chrom': 0.3,
            'pos': 0.3,
            'ica': 0.2
        }
        
        # Calculate weighted average
        total_weight = 0
        weighted_sum = 0
        valid_hrs = []
        
        for algo, hr in self.heart_rates.items():
            if 45 <= hr <= 200:  # Valid range
                weighted_sum += hr * weights[algo]
                total_weight += weights[algo]
                valid_hrs.append(hr)
        
        if total_weight > 0:
            # Calculate weighted average
            ensemble_hr = weighted_sum / total_weight
            
            # Check consistency
            if valid_hrs:
                std_dev = np.std(valid_hrs)
                if std_dev < 10:  # High agreement
                    self.confidence_score = min(95, 100 - std_dev * 5)
                else:
                    self.confidence_score = max(50, 100 - std_dev * 3)
            
            self.final_heart_rate = int(ensemble_hr)
        else:
            self.final_heart_rate = 0
            self.confidence_score = 0
    
    def calculate_confidence(self):
        """Calculate confidence score based on signal quality"""
        if self.final_heart_rate == 0:
            return 0
        
        # Factors affecting confidence
        factors = []
        
        # Factor 1: Signal stability
        if len(self.signal_buffers['green']) > 60:
            recent_signal = list(self.signal_buffers['green'])[-60:]
            signal_std = np.std(recent_signal)
            mean_signal = np.mean(recent_signal)
            
            if mean_signal != 0:
                cv = signal_std / mean_signal  # Coefficient of variation
                stability_score = max(0, 100 - cv * 100)
                factors.append(stability_score)
        
        # Factor 2: Heart rate stability
        if hasattr(self, 'hr_history'):
            if len(self.hr_history) > 10:
                hr_std = np.std(self.hr_history[-10:])
                hr_stability = max(0, 100 - hr_std * 2)
                factors.append(hr_stability)
        
        # Factor 3: Algorithm agreement (for ensemble)
        if self.algorithm.get() == 'ensemble':
            valid_hrs = [hr for hr in self.heart_rates.values() if 45 <= hr <= 200]
            if len(valid_hrs) > 1:
                agreement = 100 - np.std(valid_hrs)
                factors.append(agreement)
        
        # Calculate overall confidence
        if factors:
            confidence = np.mean(factors)
            return min(100, max(0, int(confidence)))
        
        return 75  # Default confidence
    
    def update_hrv_metrics(self):
        """Calculate HRV metrics from R-R intervals"""
        if self.final_heart_rate > 0:
            # Calculate R-R interval (in ms)
            rr_interval = 60000 / self.final_heart_rate
            self.rr_intervals.append(rr_interval)
            
            if len(self.rr_intervals) >= 10:
                rr_array = np.array(self.rr_intervals)
                
                # SDNN - Standard deviation of NN intervals
                self.hrv_metrics['SDNN'] = round(np.std(rr_array), 1)
                
                # RMSSD - Root mean square of successive differences
                differences = np.diff(rr_array)
                self.hrv_metrics['RMSSD'] = round(np.sqrt(np.mean(differences**2)), 1)
                
                # pNN50 - Percentage of successive intervals differing by >50ms
                nn50 = np.sum(np.abs(differences) > 50)
                self.hrv_metrics['pNN50'] = round((nn50 / len(differences)) * 100, 1)
    
    def update_plots(self):
        """Update all plots with current data"""
        # Update signal plot
        if len(self.timestamps) > 1:
            self.signal_ax.clear()
            self.signal_ax.set_facecolor('#0f0f0f')
            
            # Plot all algorithm signals
            colors = {'green': '#00ff00', 'chrom': '#ff00ff', 
                        'pos': '#00ffff', 'ica': '#ffff00'}
            
            time_axis = np.arange(len(self.timestamps))
            
            for algo, buffer in self.signal_buffers.items():
                if len(buffer) == len(time_axis):
                    # Normalize for display
                    signal_data = np.array(list(buffer))
                    if np.std(signal_data) > 0:
                        normalized = (signal_data - np.mean(signal_data)) / np.std(signal_data)
                        self.signal_ax.plot(time_axis, normalized, 
                                            color=colors[algo], alpha=0.7, 
                                            label=algo.upper(), linewidth=1)
            
            self.signal_ax.set_title('PPG Signals', color='white', fontsize=12)
            self.signal_ax.set_xlabel('Samples', color='white', fontsize=10)
            self.signal_ax.set_ylabel('Normalized Amplitude', color='white', fontsize=10)
            self.signal_ax.tick_params(colors='white', labelsize=8)
            self.signal_ax.grid(True, alpha=0.3)
            self.signal_ax.legend(loc='upper right', fontsize=8)
            
            self.signal_canvas.draw()
        
        # Update spectrum plot
        if self.final_heart_rate > 0 and len(self.signal_buffers['green']) > 60:
            self.spectrum_ax.clear()
            self.spectrum_ax.set_facecolor('#0f0f0f')
            
            # Get signal
            signal_data = np.array(list(self.signal_buffers['green']))
            
            # Compute spectrum
            freqs, psd = signal.welch(signal_data, fs=self.current_fps, 
                                    nperseg=min(len(signal_data), 256))
            
            # Convert to BPM
            bpm_freqs = freqs * 60
            
            # Plot
            self.spectrum_ax.plot(bpm_freqs, psd, 'g-', linewidth=2)
            
            # Mark detected HR
            self.spectrum_ax.axvline(x=self.final_heart_rate, color='r', 
                                    linestyle='--', alpha=0.8, 
                                    label=f'HR: {self.final_heart_rate} BPM')
            
            self.spectrum_ax.set_xlim(30, 200)
            self.spectrum_ax.set_title('Frequency Spectrum', color='white', fontsize=12)
            self.spectrum_ax.set_xlabel('Heart Rate (BPM)', color='white', fontsize=10)
            self.spectrum_ax.set_ylabel('Power Spectral Density', color='white', fontsize=10)
            self.spectrum_ax.tick_params(colors='white', labelsize=8)
            self.spectrum_ax.grid(True, alpha=0.3)
            self.spectrum_ax.legend(loc='upper right', fontsize=8)
            
            self.spectrum_canvas.draw()
    
    def update_statistics(self):
        """Update session statistics"""
        if hasattr(self, 'session_hrs') and self.session_hrs:
            self.stats_labels['Average HR'].config(text=f"{int(np.mean(self.session_hrs))} BPM")
            self.stats_labels['Min HR'].config(text=f"{int(np.min(self.session_hrs))} BPM")
            self.stats_labels['Max HR'].config(text=f"{int(np.max(self.session_hrs))} BPM")
            self.stats_labels['HR Variability'].config(text=f"{int(np.std(self.session_hrs))} BPM")
            
            if self.confidence_score > 0:
                quality = "Excellent" if self.confidence_score > 85 else \
                            "Good" if self.confidence_score > 70 else \
                            "Fair" if self.confidence_score > 50 else "Poor"
                self.stats_labels['Signal Quality'].config(text=quality)
    
    def animate_heart(self):
        """Animate heart icon based on BPM"""
        if self.is_running and self.final_heart_rate > 0:
            # Calculate animation speed based on heart rate
            interval = 60000 / self.final_heart_rate  # ms per beat
            
            # Pulse effect
            current_size = self.heart_label.cget('font')[1]
            new_size = 45 if current_size == 40 else 40
            self.heart_label.config(font=('Arial', new_size))
            
            self.root.after(int(interval), self.animate_heart)
    
    def update_video_display(self, frame):
        """Update video display with frame"""
        from PIL import Image, ImageTk
        
        # Resize frame
        height, width = frame.shape[:2]
        scale = min(640/width, 480/height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        resized = cv2.resize(frame, (new_width, new_height))
        
        # Add overlay information
        if self.is_running:
            # Add FPS
            if self.fps_buffer:
                avg_fps = np.mean(list(self.fps_buffer))
                cv2.putText(resized, f"FPS: {int(avg_fps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add status
            cv2.putText(resized, self.status, (10, new_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Convert and display
        image = Image.fromarray(resized)
        photo = ImageTk.PhotoImage(image=image)
        
        self.video_label.configure(image=photo)
        self.video_label.image = photo
    
    def update_status(self):
        """Update all status displays"""
        self.status_label.config(text=self.status)
        
        # Update FPS
        if self.fps_buffer:
            avg_fps = np.mean(list(self.fps_buffer))
            self.fps_label.config(text=f"FPS: {int(avg_fps)}")
    
    def update_gui(self):
        """Main GUI update loop"""
        if self.is_running:
            # Update video
            try:
                frame = self.result_queue.get_nowait()
                self.update_video_display(frame)
            except queue.Empty:
                pass
            
            # Update displays
            self.update_status()
            
            # Update heart rate display
            if self.final_heart_rate > 0:
                self.hr_label.config(text=f"{self.final_heart_rate} BPM")
                self.confidence_label.config(text=f"{self.confidence_score}%")
                
                # Update algorithm displays
                for algo, label in self.algo_labels.items():
                    hr = self.heart_rates.get(algo, 0)
                    if hr > 0:
                        label.config(text=f"{hr} BPM")
                    else:
                        label.config(text="-- BPM")
                
                # Update HRV displays
                for metric, label in self.hrv_labels.items():
                    value = self.hrv_metrics.get(metric, 0)
                    if metric == 'pNN50':
                        label.config(text=f"{value}%")
                    else:
                        label.config(text=f"{value} ms")
                
                # Store HR for session stats
                if not hasattr(self, 'session_hrs'):
                    self.session_hrs = []
                self.session_hrs.append(self.final_heart_rate)
                
                # Update plots and stats
                self.update_plots()
                self.update_statistics()
            
            # Schedule next update
            self.root.after(100, self.update_gui)
    
    def save_recording(self):
        """Save recorded data to file"""
        if not self.recorded_data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rppg_recording_{timestamp}.json"
        
        duration = time.time() - self.recording_start_time
        
        data = {
            'timestamp': timestamp,
            'duration': duration,
            'algorithm': self.algorithm.get(),
            'data': self.recorded_data,
            'statistics': {
                'average_hr': np.mean([d['heart_rate'] for d in self.recorded_data]),
                'min_hr': np.min([d['heart_rate'] for d in self.recorded_data]),
                'max_hr': np.max([d['heart_rate'] for d in self.recorded_data]),
                'std_hr': np.std([d['heart_rate'] for d in self.recorded_data]),
                'average_confidence': np.mean([d['confidence'] for d in self.recorded_data])
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        messagebox.showinfo("Recording Saved", f"Recording saved to {filename}")
    
    def save_session(self):
        """Save session data for history"""
        if not hasattr(self, 'session_hrs') or not self.session_hrs:
            return
        
        timestamp = datetime.now()
        session_data = {
            'timestamp': timestamp.isoformat(),
            'date': timestamp.strftime("%Y-%m-%d %H:%M"),
            'duration': len(self.session_hrs) / self.current_fps,
            'average_hr': int(np.mean(self.session_hrs)),
            'min_hr': int(np.min(self.session_hrs)),
            'max_hr': int(np.max(self.session_hrs)),
            'algorithm': self.algorithm.get(),
            'hrv_metrics': self.hrv_metrics.copy(),
            'confidence': self.confidence_score
        }
        
        # Load existing history
        history_file = 'rppg_history.json'
        history = []
        
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    history = json.load(f)
            except:
                history = []
        
        # Add new session
        history.append(session_data)
        
        # Keep only last 100 sessions
        history = history[-100:]
        
        # Save updated history
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Update history display
        self.load_history()
    
    def load_history(self):
        """Load and display session history"""
        history_file = 'rppg_history.json'
        
        if not os.path.exists(history_file):
            return
        
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Clear existing items
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            
            # Add history items
            for i, session in enumerate(reversed(history[-50:])):  # Show last 50
                duration_min = session['duration'] / 60
                self.history_tree.insert('', 'end', 
                                        text=f"Session {len(history)-i}",
                                        values=(
                                            session['date'],
                                            f"{duration_min:.1f} min",
                                            f"{session['average_hr']} BPM",
                                            session['algorithm'].upper()
                                        ))
        except Exception as e:
            print(f"Error loading history: {e}")
    
    def load_session(self):
        """Load a saved session file"""
        filename = filedialog.askopenfilename(
            title="Select session file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r') as f:
                    data = json.load(f)
                
                # Display session info
                info = f"Session from: {data.get('timestamp', 'Unknown')}\n"
                info += f"Duration: {data.get('duration', 0):.1f} seconds\n"
                info += f"Algorithm: {data.get('algorithm', 'Unknown')}\n"
                
                if 'statistics' in data:
                    stats = data['statistics']
                    info += f"\nAverage HR: {stats.get('average_hr', 0):.0f} BPM\n"
                    info += f"Min HR: {stats.get('min_hr', 0):.0f} BPM\n"
                    info += f"Max HR: {stats.get('max_hr', 0):.0f} BPM\n"
                    info += f"Confidence: {stats.get('average_confidence', 0):.0f}%"
                
                messagebox.showinfo("Session Info", info)
                
                # Plot session data if available
                if 'data' in data and data['data']:
                    self.plot_session_data(data['data'])
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load session: {e}")
    
    def plot_session_data(self, session_data):
        """Plot loaded session data"""
        # Create new window for session plot
        plot_window = tk.Toplevel(self.root)
        plot_window.title("Session Data Visualization")
        plot_window.geometry("1000x600")
        plot_window.configure(bg='#0f0f0f')
        
        # Create figure
        fig = Figure(figsize=(12, 8), dpi=80)
        fig.patch.set_facecolor('#0f0f0f')
        
        # Heart rate plot
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_facecolor('#1a1a1a')
        
        timestamps = [d['timestamp'] - session_data[0]['timestamp'] for d in session_data]
        heart_rates = [d['heart_rate'] for d in session_data]
        confidences = [d['confidence'] for d in session_data]
        
        ax1.plot(timestamps, heart_rates, 'r-', linewidth=2, label='Heart Rate')
        ax1.set_xlabel('Time (seconds)', color='white')
        ax1.set_ylabel('Heart Rate (BPM)', color='white')
        ax1.set_title('Heart Rate Over Time', color='white', fontsize=14)
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Confidence plot
        ax2 = ax1.twinx()
        ax2.plot(timestamps, confidences, 'g--', alpha=0.7, label='Confidence')
        ax2.set_ylabel('Confidence (%)', color='green')
        ax2.tick_params(axis='y', colors='green')
        
        # Algorithm comparison plot
        ax3 = fig.add_subplot(2, 1, 2)
        ax3.set_facecolor('#1a1a1a')
        
        # Extract algorithm data
        algos = ['green', 'chrom', 'pos', 'ica']
        colors = ['#00ff00', '#ff00ff', '#00ffff', '#ffff00']
        
        for algo, color in zip(algos, colors):
            algo_hrs = [d['algorithms'].get(algo, 0) for d in session_data]
            if any(hr > 0 for hr in algo_hrs):
                ax3.plot(timestamps, algo_hrs, color=color, alpha=0.7, label=algo.upper())
        
        ax3.set_xlabel('Time (seconds)', color='white')
        ax3.set_ylabel('Heart Rate (BPM)', color='white')
        ax3.set_title('Algorithm Comparison', color='white', fontsize=14)
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Add canvas
        canvas = FigureCanvasTkAgg(fig, plot_window)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        canvas.draw()
    
    def clear_history(self):
        """Clear session history"""
        result = messagebox.askyesno("Clear History", 
                                    "Are you sure you want to clear all session history?")
        if result:
            try:
                if os.path.exists('rppg_history.json'):
                    os.remove('rppg_history.json')
                
                # Clear tree
                for item in self.history_tree.get_children():
                    self.history_tree.delete(item)
                
                messagebox.showinfo("Success", "History cleared successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to clear history: {e}")
    
    def export_data(self):
        """Export current session data"""
        if not hasattr(self, 'session_hrs') or not self.session_hrs:
            messagebox.showwarning("No Data", "No session data to export")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                if filename.endswith('.csv'):
                    # Export as CSV
                    import csv
                    with open(filename, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Sample', 'Heart Rate (BPM)', 'Confidence (%)'])
                        
                        for i, hr in enumerate(self.session_hrs):
                            writer.writerow([i, hr, self.confidence_score])
                else:
                    # Export as JSON
                    data = {
                        'timestamp': datetime.now().isoformat(),
                        'algorithm': self.algorithm.get(),
                        'heart_rates': self.session_hrs,
                        'statistics': {
                            'average': np.mean(self.session_hrs),
                            'min': np.min(self.session_hrs),
                            'max': np.max(self.session_hrs),
                            'std': np.std(self.session_hrs)
                        },
                        'hrv_metrics': self.hrv_metrics,
                        'settings': {
                            'roi_size': self.roi_size.get(),
                            'filter_order': self.filter_order.get(),
                            'window_size': self.window_size.get()
                        }
                    }
                    
                    with open(filename, 'w') as f:
                        json.dump(data, f, indent=2)
                
                messagebox.showinfo("Export Complete", f"Data exported to {filename}")
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export data: {e}")
    
    def generate_report(self):
        """Generate a comprehensive PDF report"""
        if not hasattr(self, 'session_hrs') or not self.session_hrs:
            messagebox.showwarning("No Data", "No session data for report")
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from datetime import datetime
            
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            
            if not filename:
                return
            
            # Create PDF
            with PdfPages(filename) as pdf:
                # Page 1: Summary
                fig = plt.figure(figsize=(8.5, 11))
                fig.patch.set_facecolor('white')
                
                # Title
                fig.text(0.5, 0.95, 'rPPG Heart Rate Analysis Report', 
                        ha='center', size=20, weight='bold')
                fig.text(0.5, 0.92, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
                        ha='center', size=12)
                
                # Summary statistics
                y_pos = 0.85
                stats_text = f"""
    Session Statistics:
    - Duration: {len(self.session_hrs) / self.current_fps:.1f} seconds
    - Average Heart Rate: {np.mean(self.session_hrs):.0f} BPM
    - Minimum Heart Rate: {np.min(self.session_hrs):.0f} BPM
    - Maximum Heart Rate: {np.max(self.session_hrs):.0f} BPM
    - Standard Deviation: {np.std(self.session_hrs):.1f} BPM
    - Algorithm Used: {self.algorithm.get().upper()}
    - Average Confidence: {self.confidence_score}%

    HRV Metrics:
    - SDNN: {self.hrv_metrics['SDNN']} ms
    - RMSSD: {self.hrv_metrics['RMSSD']} ms
    - pNN50: {self.hrv_metrics['pNN50']}%
                """
                
                fig.text(0.1, y_pos, stats_text, size=12, verticalalignment='top')
                
                # Heart rate plot
                ax1 = fig.add_subplot(2, 1, 1, position=[0.1, 0.35, 0.8, 0.25])
                time_axis = np.arange(len(self.session_hrs)) / self.current_fps
                ax1.plot(time_axis, self.session_hrs, 'r-', linewidth=2)
                ax1.set_xlabel('Time (seconds)')
                ax1.set_ylabel('Heart Rate (BPM)')
                ax1.set_title('Heart Rate Over Time')
                ax1.grid(True, alpha=0.3)
                
                # Heart rate distribution
                ax2 = fig.add_subplot(2, 1, 2, position=[0.1, 0.05, 0.8, 0.25])
                ax2.hist(self.session_hrs, bins=20, color='skyblue', edgecolor='black')
                ax2.set_xlabel('Heart Rate (BPM)')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Heart Rate Distribution')
                ax2.grid(True, alpha=0.3, axis='y')
                
                pdf.savefig(fig)
                plt.close(fig)
                
                # Page 2: Algorithm comparison (if ensemble mode)
                if self.algorithm.get() == 'ensemble' and hasattr(self, 'algo_history'):
                    fig2 = plt.figure(figsize=(8.5, 11))
                    fig2.patch.set_facecolor('white')
                    
                    fig2.text(0.5, 0.95, 'Algorithm Comparison', 
                                ha='center', size=18, weight='bold')
                    
                    # Plot each algorithm
                    ax = fig2.add_subplot(1, 1, 1, position=[0.1, 0.1, 0.8, 0.8])
                    
                    colors = {'green': '#00ff00', 'chrom': '#ff00ff', 
                                'pos': '#00ffff', 'ica': '#ffff00'}
                    
                    for algo, color in colors.items():
                        if algo in self.algo_history:
                            ax.plot(time_axis[:len(self.algo_history[algo])], 
                                    self.algo_history[algo], 
                                    color=color, label=algo.upper(), alpha=0.7)
                    
                    ax.set_xlabel('Time (seconds)')
                    ax.set_ylabel('Heart Rate (BPM)')
                    ax.set_title('Algorithm Comparison')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    pdf.savefig(fig2)
                    plt.close(fig2)
            
            messagebox.showinfo("Report Generated", f"Report saved to {filename}")
            
        except ImportError:
            messagebox.showerror("Error", "matplotlib is required for PDF generation")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")


def main():
   """Main entry point"""
   root = tk.Tk()
   
   # Set icon if available
   try:
       root.iconbitmap('heart.ico')
   except:
       pass
   
   # Create application
   app = AdvancedRPPGApplication(root)
   
   # Center window
   root.update_idletasks()
   width = root.winfo_width()
   height = root.winfo_height()
   x = (root.winfo_screenwidth() // 2) - (width // 2)
   y = (root.winfo_screenheight() // 2) - (height // 2)
   root.geometry(f'{width}x{height}+{x}+{y}')
   
   # Run
   root.mainloop()


if __name__ == "__main__":
   main()