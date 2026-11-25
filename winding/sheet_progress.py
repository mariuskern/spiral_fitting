"""Portable sheet pipeline: one progress bar and warning relay for all workers."""

from contextlib import contextmanager
import logging
from logging.handlers import QueueHandler, QueueListener
import sys

from tqdm import tqdm


LOGGER_NAME = "portable_sheet"


class _ProgressWarnings(logging.Handler):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record):
        tqdm.write(self.format(record), file=self.stream)


class _ForwardWarnings(logging.Handler):
    def emit(self, record):
        logging.getLogger(LOGGER_NAME).handle(record)


@contextmanager
def block_progress(total, device, workers, disabled=False):
    """Count committed blocks; restore caller logging configuration on exit."""
    logger = logging.getLogger(LOGGER_NAME)
    previous = logger.handlers[:], logger.level, logger.propagate
    handler = _ProgressWarnings(sys.stderr)
    logger.handlers = [handler]
    logger.setLevel(logging.WARNING)
    logger.propagate = False
    try:
        with tqdm(total=total, desc=f"Blocks ({device}, workers={workers})",
                  unit="block", dynamic_ncols=True, ascii=True, mininterval=0.2,
                  file=sys.stderr, disable=disabled) as progress:
            try:
                yield progress
            except BaseException:
                progress.set_description("Stopped", refresh=True)
                raise
    finally:
        logger.handlers, level, logger.propagate = previous
        logger.setLevel(level)
        handler.close()


def configure_worker_warnings(queue):
    """Worker processes send warnings, never terminal output, to the parent."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers = [QueueHandler(queue)]
    logger.setLevel(logging.WARNING)
    logger.propagate = False


@contextmanager
def worker_warnings(context):
    queue = context.Queue()
    listener = QueueListener(queue, _ForwardWarnings())
    listener.start()
    try:
        yield queue
    finally:
        listener.stop()
        queue.close()
        queue.join_thread()
