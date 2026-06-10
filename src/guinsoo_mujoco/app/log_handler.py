from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal


class QtLogHandler(logging.Handler, QObject):
    log_record = Signal(str)

    def __init__(self, level: int = logging.NOTSET) -> None:
        logging.Handler.__init__(self, level)
        QObject.__init__(self)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        self.log_record.emit(message)
