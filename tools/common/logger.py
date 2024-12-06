import logging


def configure_logger(log_file="app.log", level=logging.DEBUG) -> logging.Logger:
    logger = logging.getLogger("Log")
    if not logger.hasHandlers():
        logger.setLevel(level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(filename)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


logger = configure_logger()
