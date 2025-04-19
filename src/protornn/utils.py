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


def get_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert string representation of PyTorch dtype to `torch.dtype` object

    Args:
        dtype: string representation of a valid torch dtype or a `torch.dtype`
            object (no-op)

    Raises:
        ValueError: if `dtype` is not a valid dtype or string representation

    Returns:
        A valid torch dtype

    Examples:
        >>> get_torch_dtype(torch.float32)
        torch.float32

        >>> get_torch_dtype('float32')
        torch.float32

        >>> get_torch_dtype('torch.Float32')
        torch.float32

        >>> get_torch_dtype('Tensor')
        ValueError: 'Tensor' is not a valid torch dtype
    """
    if isinstance(dtype, str):
        dtype_str = dtype.lower().removeprefix("torch.")
        try:
            attr = getattr(torch, dtype_str)
            if isinstance(attr, torch.dtype):
                return attr
            else:
                raise AttributeError
        except AttributeError:
            raise ValueError(f"'{dtype}' is not a valid torch dtype")

    elif isinstance(dtype, torch.dtype):
        return dtype

    else:
        raise ValueError(f"Expected torch.dtype or string, got {type(dtype)}")


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
