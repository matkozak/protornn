import logging
from os import PathLike

import torch


def get_device() -> torch.device:
    """CUDA or MPS if available, CPU otherwise."""
    if torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    return torch.device(dev)


def setup_logging(
    logfile: str | PathLike | None = None,
    stream_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> None:
    """Configure root logger with consistent formatting and handlers.

    Sets up the root logger with a StreamHandler for console output and
    optionally a FileHandler for detailed logging. Both handlers use the
    same timestamp and module-aware format.

    Args:
        logfile: Optional path to write detailed logs. If provided, logs
            will be written to this file in addition to console output.
            The file will be created or overwritten.
        stream_level: Logging level for console output
        file_level: Logging level for file output

    Example:
        >>> setup_logging()  # Console logging only at INFO level
        >>> setup_logging("process.log")  # Console + file logging
        >>> setup_logging("debug.log", logging.WARNING, logging.DEBUG)
    """
    logger = logging.getLogger()
    logger.setLevel(min(stream_level, file_level))

    # Formatter
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Add stream handler
    sh = logging.StreamHandler()
    sh.setLevel(stream_level)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Add file handler
    if logfile is not None:
        fh = logging.FileHandler(logfile, mode="w")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
