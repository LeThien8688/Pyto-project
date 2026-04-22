# CNC_Gcode

Bộ công cụ Python xử lý G-code cho máy CNC, chạy được trên **Pyto** (iOS/iPadOS).

## Cấu trúc

```
CNC_Gcode/
├── README.md          # Tài liệu
└── gcode_parser.py    # Đọc & phân tích file G-code
```

## Yêu cầu

- Pyto (iOS) hoặc Python 3.8+
- Không cần thư viện ngoài cho `gcode_parser.py`

## Cách dùng

### Parse một file G-code

```python
from gcode_parser import GcodeParser

parser = GcodeParser()
commands = parser.parse_file("examples/sample.gcode")

for cmd in commands:
    print(cmd)
```

### Parse một chuỗi G-code

```python
from gcode_parser import GcodeParser

gcode = """
G21           ; milimet
G90           ; toạ độ tuyệt đối
G0 X0 Y0 Z5
G1 X10 Y10 F200
M2            ; kết thúc
"""

parser = GcodeParser()
commands = parser.parse_text(gcode)
print(f"Tổng số lệnh: {len(commands)}")
```

## Các lệnh G-code hỗ trợ

Parser nhận diện được mọi lệnh dạng `<letter><number>` phổ biến:

| Nhóm | Ví dụ | Ý nghĩa |
|------|-------|---------|
| G    | G0, G1, G2, G3, G21, G90, G91 | Lệnh chuyển động / chế độ |
| M    | M2, M3, M5, M30 | Lệnh máy (spindle, kết thúc...) |
| T    | T1, T2 | Chọn dao |
| X Y Z | X10.5 Y-3 Z0.2 | Toạ độ |
| F    | F1200 | Tốc độ chạy (feed rate) |
| S    | S8000 | Tốc độ spindle |
| I J K R | I5 J0 | Tham số cung tròn |

Comment được hỗ trợ cả 2 kiểu: `; comment` và `( comment )`.

## Kế hoạch mở rộng

- [ ] `gcode_visualizer.py` — vẽ đường chạy dao (matplotlib)
- [ ] `gcode_generator.py` — sinh G-code từ hình cơ bản
- [ ] `gcode_sender.py` — gửi qua WiFi tới GRBL/FluidNC
