import torch
import platform
import psutil

print("🧠 시스템 정보 확인")

# CPU 정보
print(f"🔸 CPU: {platform.processor()}")
print(f"🔸 CPU 코어 수: {psutil.cpu_count(logical=False)} (물리), {psutil.cpu_count(logical=True)} (논리)")

# RAM 정보
ram = psutil.virtual_memory()
print(f"🔸 RAM: {round(ram.total / (1024**3), 2)} GB")

# GPU 정보 (PyTorch 기준)
if torch.cuda.is_available():
    print("🔹 GPU 사용 가능 ✅")
    print(f"🔹 GPU 이름: {torch.cuda.get_device_name(0)}")
    print(torch.cuda.get_device_name(0))
    print(f"🔹 GPU 메모리: {round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)} GB")
else:
    print("🔹 GPU 사용 불가 ❌ (torch.cuda.is_available() = False)")
