"""
G-code parser — đọc và phân tích file/chuỗi G-code.

Chạy trực tiếp trên Pyto (iOS) — không cần thư viện ngoài.

Ví dụ:
    from gcode_parser import GcodeParser

    parser = GcodeParser()
    commands = parser.parse_file("examples/sample.gcode")
    for cmd in commands:
        print(cmd)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Regex bắt các token dạng: chữ cái + số (có thể âm, thập phân)
# Ví dụ: G0, X-10.5, F1200, S8000
_TOKEN_RE = re.compile(r"([A-Za-z])\s*(-?\d+(?:\.\d+)?)")

# Comment kiểu ( ... ) — bỏ qua nội dung bên trong
_PAREN_COMMENT_RE = re.compile(r"\([^)]*\)")


@dataclass
class GcodeCommand:
    """Một dòng G-code đã được parse."""

    line_number: int
    raw: str
    command: str | None = None            # "G1", "M3", "T1"... (lệnh chính của dòng)
    params: dict[str, float] = field(default_factory=dict)  # {"X": 10.5, "Y": -3, "F": 200}
    comment: str = ""

    def __str__(self) -> str:
        parts = []
        if self.command:
            parts.append(self.command)
        for key, value in self.params.items():
            if value == int(value):
                parts.append(f"{key}{int(value)}")
            else:
                parts.append(f"{key}{value}")
        body = " ".join(parts) if parts else "(empty)"
        if self.comment:
            return f"[{self.line_number}] {body}  ; {self.comment}"
        return f"[{self.line_number}] {body}"


class GcodeParser:
    """Parser G-code đơn giản, đủ dùng cho CNC nhỏ (GRBL, Marlin, FluidNC...)."""

    # Các chữ cái được coi là "lệnh chính" (nếu xuất hiện đầu dòng)
    COMMAND_LETTERS = {"G", "M", "T"}

    def parse_text(self, text: str) -> list[GcodeCommand]:
        """Parse một chuỗi G-code, trả về list các GcodeCommand."""
        commands: list[GcodeCommand] = []
        for idx, raw_line in enumerate(text.splitlines(), start=1):
            cmd = self._parse_line(raw_line, idx)
            if cmd is not None:
                commands.append(cmd)
        return commands

    def parse_file(self, path: str, encoding: str = "utf-8") -> list[GcodeCommand]:
        """Parse một file G-code."""
        with open(path, "r", encoding=encoding) as f:
            return self.parse_text(f.read())

    def _parse_line(self, raw: str, line_number: int) -> GcodeCommand | None:
        original = raw.rstrip("\n\r")
        line = original

        # Tách comment
        comment = ""
        if ";" in line:
            line, _, after = line.partition(";")
            comment = after.strip()

        paren_comments = _PAREN_COMMENT_RE.findall(line)
        if paren_comments:
            extra = " ".join(c.strip("()").strip() for c in paren_comments)
            comment = f"{comment} {extra}".strip() if comment else extra
            line = _PAREN_COMMENT_RE.sub(" ", line)

        line = line.strip()

        # Dòng trống hoặc chỉ có comment → vẫn trả về để giữ số dòng nếu có comment,
        # còn dòng trống hẳn thì bỏ qua.
        if not line and not comment:
            return None

        command: str | None = None
        params: dict[str, float] = {}

        for letter, number in _TOKEN_RE.findall(line):
            letter_up = letter.upper()
            value = float(number)

            if command is None and letter_up in self.COMMAND_LETTERS:
                # Giữ dạng số nguyên nếu có thể: "G1" thay vì "G1.0"
                if value == int(value):
                    command = f"{letter_up}{int(value)}"
                else:
                    command = f"{letter_up}{value}"
            else:
                params[letter_up] = value

        return GcodeCommand(
            line_number=line_number,
            raw=original,
            command=command,
            params=params,
            comment=comment,
        )


def _demo() -> None:
    """Chạy thử nhanh: python gcode_parser.py"""
    sample = """
    ; Demo G-code
    G21                 ; milimet
    G90                 ; absolute
    G0 X0 Y0 Z5
    M3 S8000            (bat spindle)
    G1 X10 Y10 F1200
    G1 Z-1
    G2 X20 Y10 I5 J0
    G0 Z5
    M5
    M30                 ; end
    """
    parser = GcodeParser()
    for cmd in parser.parse_text(sample):
        print(cmd)


if __name__ == "__main__":
    _demo()
