#!/usr/bin/env python3
"""
CyberPuppy Resource Monitor
==========================

Monitors system resources during training and sends alerts.
"""

import argparse
import json
import os
import sys
import time
import psutil
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import logging


class ResourceMonitor:
    """Monitor system resources during training with alerts."""

    def __init__(self, session_id: str, project_root: Optional[str] = None):
        self.session_id = session_id
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.monitoring = False
        self.monitor_thread = None

        # Resource thresholds for alerts
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'gpu_memory_percent': 90.0,  # If GPU available
            'temperature': 80.0  # If temperature sensors available
        }

        # Monitoring configuration
        self.monitor_interval = 30  # seconds
        self.alert_cooldown = 300   # 5 minutes between similar alerts
        self.last_alerts = {}

        # Setup logging
        self.setup_logging()

        # Resource history for analysis
        self.resource_history = []
        self.max_history_size = 1000

    def setup_logging(self):
        """Setup logging for the resource monitor."""
        log_file = self.logs_dir / f"resource_monitor_{self.session_id}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger(__name__)

    def get_system_info(self) -> Dict:
        """Get comprehensive system information."""
        try:
            # CPU information
            cpu_info = {
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'cpu_percent': psutil.cpu_percent(interval=1, percpu=True),
                'cpu_percent_avg': psutil.cpu_percent(interval=1)
            }

            # Memory information
            memory = psutil.virtual_memory()
            memory_info = {
                'total': memory.total,
                'available': memory.available,
                'percent': memory.percent,
                'used': memory.used,
                'free': memory.free
            }

            # Disk information
            disk = psutil.disk_usage('/')
            disk_info = {
                'total': disk.total,
                'used': disk.used,
                'free': disk.free,
                'percent': disk.percent
            }

            # Network information
            network = psutil.net_io_counters()
            network_info = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }

            # Process information
            process_info = {
                'total_processes': len(psutil.pids()),
                'python_processes': len([p for p in psutil.process_iter(['name']) if 'python' in p.info['name'].lower()])
            }

            # GPU information (if available)
            gpu_info = self.get_gpu_info()

            # Temperature information (if available)
            temp_info = self.get_temperature_info()

            return {
                'timestamp': datetime.now().isoformat(),
                'cpu': cpu_info,
                'memory': memory_info,
                'disk': disk_info,
                'network': network_info,
                'processes': process_info,
                'gpu': gpu_info,
                'temperature': temp_info
            }

        except Exception as e:
            self.logger.error(f"Error getting system info: {e}")
            return {}

    def get_gpu_info(self) -> Dict:
        """Get GPU information if available."""
        gpu_info = {'available': False}

        try:
            import GPUtil
            gpus = GPUtil.getGPUs()

            if gpus:
                gpu_info['available'] = True
                gpu_info['gpus'] = []

                for i, gpu in enumerate(gpus):
                    gpu_data = {
                        'id': i,
                        'name': gpu.name,
                        'load': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    }
                    gpu_info['gpus'].append(gpu_data)

        except ImportError:
            # GPUtil not available
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                device_count = nvml.nvmlDeviceGetCount()

                if device_count > 0:
                    gpu_info['available'] = True
                    gpu_info['gpus'] = []

                    for i in range(device_count):
                        handle = nvml.nvmlDeviceGetHandleByIndex(i)
                        name = nvml.nvmlDeviceGetName(handle).decode()
                        memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                        gpu_data = {
                            'id': i,
                            'name': name,
                            'memory_used': memory_info.used // 1024**2,  # MB
                            'memory_total': memory_info.total // 1024**2,  # MB
                            'memory_percent': (memory_info.used / memory_info.total) * 100
                        }

                        try:
                            utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_data['load'] = utilization.gpu
                        except:
                            gpu_data['load'] = 0

                        try:
                            temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                            gpu_data['temperature'] = temp
                        except:
                            gpu_data['temperature'] = None

                        gpu_info['gpus'].append(gpu_data)

            except ImportError:
                pass
            except Exception as e:
                self.logger.warning(f"Error getting GPU info: {e}")

        return gpu_info

    def get_temperature_info(self) -> Dict:
        """Get temperature information if available."""
        temp_info = {'available': False}

        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    temp_info['available'] = True
                    temp_info['sensors'] = {}

                    for name, entries in temps.items():
                        temp_info['sensors'][name] = []
                        for entry in entries:
                            sensor_data = {
                                'label': entry.label or 'Unknown',
                                'current': entry.current,
                                'high': entry.high,
                                'critical': entry.critical
                            }
                            temp_info['sensors'][name].append(sensor_data)

        except Exception as e:
            self.logger.warning(f"Error getting temperature info: {e}")

        return temp_info

    def check_thresholds(self, system_info: Dict) -> List[Dict]:
        """Check if any resource thresholds are exceeded."""
        alerts = []
        current_time = datetime.now()

        # Check CPU
        cpu_avg = system_info.get('cpu', {}).get('cpu_percent_avg', 0)
        if cpu_avg > self.thresholds['cpu_percent']:
            alert_key = 'cpu_high'
            if self.should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'cpu_high',
                    'severity': 'warning',
                    'message': f'High CPU usage: {cpu_avg:.1f}%',
                    'value': cpu_avg,
                    'threshold': self.thresholds['cpu_percent']
                })

        # Check Memory
        memory_percent = system_info.get('memory', {}).get('percent', 0)
        if memory_percent > self.thresholds['memory_percent']:
            alert_key = 'memory_high'
            if self.should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'memory_high',
                    'severity': 'warning',
                    'message': f'High memory usage: {memory_percent:.1f}%',
                    'value': memory_percent,
                    'threshold': self.thresholds['memory_percent']
                })

        # Check Disk
        disk_percent = system_info.get('disk', {}).get('percent', 0)
        if disk_percent > self.thresholds['disk_percent']:
            alert_key = 'disk_high'
            if self.should_send_alert(alert_key, current_time):
                alerts.append({
                    'type': 'disk_high',
                    'severity': 'critical',
                    'message': f'High disk usage: {disk_percent:.1f}%',
                    'value': disk_percent,
                    'threshold': self.thresholds['disk_percent']
                })

        # Check GPU
        gpu_info = system_info.get('gpu', {})
        if gpu_info.get('available'):
            for gpu in gpu_info.get('gpus', []):
                memory_percent = gpu.get('memory_percent', 0)
                if memory_percent > self.thresholds['gpu_memory_percent']:
                    alert_key = f"gpu_{gpu.get('id', 0)}_memory_high"
                    if self.should_send_alert(alert_key, current_time):
                        alerts.append({
                            'type': 'gpu_memory_high',
                            'severity': 'warning',
                            'message': f"GPU {gpu.get('id', 0)} ({gpu.get('name', 'Unknown')}) high memory: {memory_percent:.1f}%",
                            'value': memory_percent,
                            'threshold': self.thresholds['gpu_memory_percent']
                        })

        # Check Temperature
        temp_info = system_info.get('temperature', {})
        if temp_info.get('available'):
            for sensor_name, sensors in temp_info.get('sensors', {}).items():
                for sensor in sensors:
                    current_temp = sensor.get('current', 0)
                    if current_temp > self.thresholds['temperature']:
                        alert_key = f"temp_{sensor_name}_{sensor.get('label', 'unknown')}"
                        if self.should_send_alert(alert_key, current_time):
                            alerts.append({
                                'type': 'temperature_high',
                                'severity': 'critical',
                                'message': f"High temperature {sensor_name} {sensor.get('label', '')}: {current_temp}°C",
                                'value': current_temp,
                                'threshold': self.thresholds['temperature']
                            })

        return alerts

    def should_send_alert(self, alert_key: str, current_time: datetime) -> bool:
        """Check if enough time has passed since the last alert of this type."""
        if alert_key not in self.last_alerts:
            self.last_alerts[alert_key] = current_time
            return True

        time_diff = current_time - self.last_alerts[alert_key]
        if time_diff.total_seconds() >= self.alert_cooldown:
            self.last_alerts[alert_key] = current_time
            return True

        return False

    def send_alerts(self, alerts: List[Dict]):
        """Send alerts for resource threshold violations."""
        if not alerts:
            return

        for alert in alerts:
            self.logger.warning(f"ALERT: {alert['message']}")

            # Log to alert file
            alert_file = self.logs_dir / f"alerts_{self.session_id}.log"
            with open(alert_file, 'a', encoding='utf-8') as f:
                alert_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': self.session_id,
                    **alert
                }
                f.write(json.dumps(alert_entry, ensure_ascii=False) + '\n')

        # Send desktop notification for critical alerts
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        if critical_alerts:
            self.send_desktop_notification(critical_alerts)

    def send_desktop_notification(self, alerts: List[Dict]):
        """Send desktop notification for critical alerts."""
        try:
            import plyer
            title = "CyberPuppy Training Alert"
            message = f"Critical resource alerts detected:\n"
            for alert in alerts[:3]:  # Limit to first 3 alerts
                message += f"• {alert['message']}\n"

            if len(alerts) > 3:
                message += f"... and {len(alerts) - 3} more alerts"

            plyer.notification.notify(
                title=title,
                message=message,
                timeout=10
            )

        except ImportError:
            self.logger.warning("plyer not available for desktop notifications")
        except Exception as e:
            self.logger.error(f"Failed to send desktop notification: {e}")

    def monitor_loop(self):
        """Main monitoring loop."""
        self.logger.info(f"Starting resource monitoring for session {self.session_id}")
        self.logger.info(f"Monitoring interval: {self.monitor_interval} seconds")
        self.logger.info(f"Alert cooldown: {self.alert_cooldown} seconds")

        while self.monitoring:
            try:
                # Get current system info
                system_info = self.get_system_info()

                if system_info:
                    # Add to history
                    self.resource_history.append(system_info)
                    if len(self.resource_history) > self.max_history_size:
                        self.resource_history.pop(0)

                    # Check thresholds and send alerts
                    alerts = self.check_thresholds(system_info)
                    if alerts:
                        self.send_alerts(alerts)

                    # Log resource summary
                    cpu_avg = system_info.get('cpu', {}).get('cpu_percent_avg', 0)
                    memory_percent = system_info.get('memory', {}).get('percent', 0)
                    disk_percent = system_info.get('disk', {}).get('percent', 0)

                    self.logger.info(
                        f"Resources: CPU {cpu_avg:.1f}%, Memory {memory_percent:.1f}%, Disk {disk_percent:.1f}%"
                    )

                    # Save resource data
                    self.save_resource_data(system_info)

                # Wait for next monitoring cycle
                time.sleep(self.monitor_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)

        self.logger.info("Resource monitoring stopped")

    def save_resource_data(self, system_info: Dict):
        """Save resource data to file for analysis."""
        try:
            resource_file = self.logs_dir / f"resources_{self.session_id}.jsonl"
            with open(resource_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(system_info, ensure_ascii=False) + '\n')

        except Exception as e:
            self.logger.error(f"Failed to save resource data: {e}")

    def start_monitoring(self):
        """Start resource monitoring in a separate thread."""
        if self.monitoring:
            self.logger.warning("Monitoring is already running")
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Resource monitoring started")

    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring:
            return

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)

        self.logger.info("Resource monitoring stopped")

        # Generate summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate a summary report of resource usage."""
        if not self.resource_history:
            return

        try:
            # Calculate statistics
            cpu_values = [info.get('cpu', {}).get('cpu_percent_avg', 0) for info in self.resource_history]
            memory_values = [info.get('memory', {}).get('percent', 0) for info in self.resource_history]
            disk_values = [info.get('disk', {}).get('percent', 0) for info in self.resource_history]

            summary = {
                'session_id': self.session_id,
                'monitoring_duration': len(self.resource_history) * self.monitor_interval,
                'cpu_stats': {
                    'min': min(cpu_values) if cpu_values else 0,
                    'max': max(cpu_values) if cpu_values else 0,
                    'avg': sum(cpu_values) / len(cpu_values) if cpu_values else 0
                },
                'memory_stats': {
                    'min': min(memory_values) if memory_values else 0,
                    'max': max(memory_values) if memory_values else 0,
                    'avg': sum(memory_values) / len(memory_values) if memory_values else 0
                },
                'disk_stats': {
                    'min': min(disk_values) if disk_values else 0,
                    'max': max(disk_values) if disk_values else 0,
                    'avg': sum(disk_values) / len(disk_values) if disk_values else 0
                },
                'total_samples': len(self.resource_history)
            }

            # Save summary
            summary_file = self.logs_dir / f"resource_summary_{self.session_id}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Resource summary saved: {summary_file}")

        except Exception as e:
            self.logger.error(f"Failed to generate summary report: {e}")


def main():
    """Main resource monitor entry point."""
    parser = argparse.ArgumentParser(description='Monitor system resources during training')
    parser.add_argument('--session-id', required=True, help='Training session ID')
    parser.add_argument('--interval', type=int, default=30, help='Monitoring interval in seconds')
    parser.add_argument('--cpu-threshold', type=float, default=90.0, help='CPU usage threshold')
    parser.add_argument('--memory-threshold', type=float, default=85.0, help='Memory usage threshold')
    parser.add_argument('--disk-threshold', type=float, default=90.0, help='Disk usage threshold')
    parser.add_argument('--project-root', help='Project root directory')

    args = parser.parse_args()

    try:
        # Initialize monitor
        monitor = ResourceMonitor(args.session_id, args.project_root)

        # Configure thresholds
        monitor.thresholds['cpu_percent'] = args.cpu_threshold
        monitor.thresholds['memory_percent'] = args.memory_threshold
        monitor.thresholds['disk_percent'] = args.disk_threshold
        monitor.monitor_interval = args.interval

        # Start monitoring
        monitor.start_monitoring()

        print(f"🔍 Resource monitoring started for session: {args.session_id}")
        print(f"   Interval: {args.interval} seconds")
        print(f"   Thresholds: CPU {args.cpu_threshold}%, Memory {args.memory_threshold}%, Disk {args.disk_threshold}%")
        print("   Press Ctrl+C to stop monitoring")

        # Keep the main thread alive
        try:
            while monitor.monitoring:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️ Stopping resource monitoring...")
            monitor.stop_monitoring()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()