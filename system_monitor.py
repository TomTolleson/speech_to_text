import psutil
import GPUtil
import threading
import time
from datetime import datetime

class SystemMonitor:
    def __init__(self):
        self.running = False
        self.monitor_thread = None
        
    def get_size(self, bytes):
        """Convert bytes to human readable format"""
        for unit in ['', 'KB', 'MB', 'GB', 'TB']:
            if bytes < 1024:
                return f"{bytes:.2f}{unit}"
            bytes /= 1024
    
    def monitor_resources(self):
        while self.running:
            # CPU Usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_used = self.get_size(memory.used)
            memory_total = self.get_size(memory.total)
            memory_percent = memory.percent
            
            # GPU Usage
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Get first GPU
                    gpu_info = f"Load: {gpu.load*100:.1f}% | Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB | Temp: {gpu.temperature}°C"
                else:
                    gpu_info = "No GPU detected"
            except Exception as e:
                gpu_info = f"Error getting GPU info: {str(e)}"
            
            # Print stats
            print("\n" + "="*50)
            print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
            print(f"CPU Usage: {cpu_usage}%")
            print(f"RAM Usage: {memory_used}/{memory_total} ({memory_percent}%)")
            print(f"GPU: {gpu_info}")
            print("="*50)
            
            time.sleep(2)
    
    def start(self):
        self.running = True
        self.monitor_thread = threading.Thread(target=self.monitor_resources)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join() 