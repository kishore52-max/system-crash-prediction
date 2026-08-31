import psutil

print("CPU Usage:", psutil.cpu_percent(), "%")
print("RAM Usage:", psutil.virtual_memory().percent, "%")
print("Disk Usage:", psutil.disk_usage("/").percent, "%")