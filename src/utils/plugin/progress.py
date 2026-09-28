import re
import time

from logging import Logger

_resolved = re.compile(r"Resolved (\d+) packages?")
_downloading = re.compile(r"Downloading (\S+) \(([\d.]+)(B|KiB|MiB|GiB)\)")
_downloaded = re.compile(r"Downloaded (\S+)")
_building = re.compile(r"Building (\S+)")
_built = re.compile(r"Built (\S+)")
_prepared = re.compile(r"Prepared (\d+) packages?")
_installed = re.compile(r"Installed (\d+) packages?")
_units = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}
_bar_width = 20
_waiting_limit = 3

def _format_size(size: float) -> str:
    for unit in ("GiB", "MiB", "KiB"):
        if size >= _units[unit]:
            return f"{size / _units[unit]:.1f} {unit}"

    return f"{size:.0f} B"

def _format_elapsed(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"

class RuntimeProgress:
    """uvの出力から、ランタイム構築の進捗をログへ出力する。"""

    def __init__(self, runtime: str, log: Logger) -> None:
        self.runtime = runtime
        self.log = log
        self.stage = "resolving dependencies"
        self.started = time.monotonic()
        self.pending: dict[str, float] = {}
        self.completed: dict[str, float] = {}

    def feed(self, line: str) -> None:
        line = line.strip()

        if match := _downloading.fullmatch(line):
            name, size, unit = match.groups()
            self.pending[name] = float(size) * _units[unit]
            self.stage = "downloading"
        elif match := _downloaded.fullmatch(line):
            name = match.group(1)
            self.completed[name] = self.pending.pop(name, 0)
            self._info(f"{self._bar()} - {name}")
        elif match := _resolved.match(line):
            self.stage = "downloading"
            self._info(f"Resolved {match.group(1)} packages")
        elif match := _building.fullmatch(line):
            self._info(f"Building {match.group(1)}")
        elif match := _built.fullmatch(line):
            self._info(f"Built {match.group(1)}")
        elif match := _prepared.match(line):
            self.stage = "installing"
            self._info(f"Prepared {match.group(1)} packages, installing")
        elif match := _installed.match(line):
            self._info(f"Installed {match.group(1)} packages")
        elif line:
            self.log.debug(f"[{self.runtime}] {line}")

    def tick(self) -> None:
        """出力が途切れている間も、処理が続いていることを示す。"""

        message = f"Still {self.stage}"

        if self.stage == "downloading" and self.pending:
            waiting = sorted(
                self.pending.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:_waiting_limit]
            names = ", ".join(
                f"{name} ({_format_size(size)})" for name, size in waiting
            )
            message = f"{self._bar()} - waiting for {names}"

        self._info(message)

    def _bar(self) -> str:
        done = sum(self.completed.values())
        total = done + sum(self.pending.values())
        ratio = done / total if total else 0
        filled = round(ratio * _bar_width)
        files = len(self.completed)
        return (
            f"Downloading [{'#' * filled}{'-' * (_bar_width - filled)}] "
            f"{ratio:4.0%} {_format_size(done)} / {_format_size(total)} "
            f"({files}/{files + len(self.pending)} files)"
        )

    def _info(self, message: str) -> None:
        elapsed = _format_elapsed(time.monotonic() - self.started)
        self.log.info(f"[{self.runtime}] {message} ({elapsed})")
