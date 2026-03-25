from __future__ import annotations

import io
import logging

from tracking_pipeline.infrastructure.logging.run_logger import get_run_logger


def test_run_logger_does_not_propagate_to_root_logger() -> None:
    logger_name = "tracking_pipeline_test_logger"
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.propagate = True

    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)
    original_level = root_logger.level

    root_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    root_handler.setFormatter(logging.Formatter("ROOT %(message)s"))
    root_logger.handlers = [root_handler]
    root_logger.setLevel(logging.INFO)

    try:
        run_logger = get_run_logger(logger_name)
        run_stream = io.StringIO()
        run_handler = logging.StreamHandler(run_stream)
        run_handler.setFormatter(logging.Formatter("%(message)s"))
        run_logger.handlers = [run_handler]

        run_logger.info("hello")

        assert run_stream.getvalue().strip() == "hello"
        assert root_stream.getvalue().strip() == ""
    finally:
        logging.getLogger(logger_name).handlers.clear()
        logging.getLogger(logger_name).propagate = True
        root_logger.handlers = original_handlers
        root_logger.setLevel(original_level)
