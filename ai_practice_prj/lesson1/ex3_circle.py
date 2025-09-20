import math

print("=== Tính chu vi và diện tích hình tròn ===")
r = float(input("Nhập bán kính r: "))

cv = 2 * math.pi * r
dt = math.pi * r**2 

print(f"Chu vi hình tròn: {cv:.2f}")
print(f"Diện tích hình tròn: {dt:.2f}")