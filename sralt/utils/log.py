import logging
import datetime
import zoneinfo

log_colors = {
    'DEBUG':    '\033[0;32m',
    'INFO':     '\033[0;36m',
    'WARNING':  '\033[0;33m',
    'ERROR':    '\033[0;31m',
    'CRITICAL': '\033[0;35m',
    'PREFIX':   '\033[0;90m',
    'RESET':    '\033[0m',
}


class LocalTZFormatter(logging.Formatter):
    def __init__(self, fmt: str, datefmt: str | None = None, tz: str | None = None):
        super().__init__(fmt, datefmt)
        self.tzinfo = zoneinfo.ZoneInfo(tz) if tz else None

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        dt = datetime.datetime.fromtimestamp(record.created)
        dt = dt.astimezone(self.tzinfo)
        return dt.strftime(datefmt or '%Y-%m-%d %H:%M:%S')


class TerminalFormatter(LocalTZFormatter):
    def format(self, record: logging.LogRecord) -> str:
        record.log_color = log_colors.get(record.levelname, log_colors['RESET'])
        record.prefix_color = log_colors['PREFIX']
        record.reset_color = log_colors['RESET']
        return super().format(record)


def setup_logging(level: int = logging.INFO, log_file: str | None = None, tz: str | None = None):
    """
    Set up logging configuration.

    :param level: Logging level (e.g., logging.DEBUG, logging.INFO).
    :param log_file: Optional file path to log to a file.
    :param tz: Timezone string (e.g., 'Asia/Shanghai'). If None, use system timezone.
    """
    terminal_format_str = '%(prefix_color)s[%(asctime)s] [%(filename)s:%(lineno)d] %(log_color)s%(message)s%(reset_color)s'
    file_format_str = '[%(asctime)s] [%(levelname)-8s] [%(filename)s:%(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(TerminalFormatter(terminal_format_str, datefmt=date_format, tz=tz))
    handlers: list[logging.Handler] = [console_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(LocalTZFormatter(file_format_str, datefmt=date_format, tz=tz))
        handlers.append(file_handler)

    logging.basicConfig(level=level, handlers=handlers, force=True)
