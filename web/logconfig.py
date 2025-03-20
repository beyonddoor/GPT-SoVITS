import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(levelname)s | %(name)s | %(asctime)s | %(message)s",
        },
    },
    # "handlers": {
    #     "file": {
    #         "class": "logging.FileHandler",
    #         "filename": "app.log",  # <-- log file name
    #         "formatter": "default",
    #     },
    #     "console": {
    #         "class": "logging.StreamHandler",
    #         "formatter": "default",
    #     },
    # },
    # "loggers": {
    #     "uvicorn": {
    #         "handlers": ["file", "console"],
    #         "level": "DEBUG",
    #         "propagate": False,
    #     },
    #     "uvicorn.error": {
    #         "handlers": ["file", "console"],
    #         "level": "DEBUG",
    #         "propagate": False,
    #     },
    #     "uvicorn.access": {
    #         "handlers": ["file", "console"],
    #         "level": "DEBUG",
    #         "propagate": False,
    #     },
    #     "main": {
    #         "handlers": ["file", "console"],
    #         "level": "DEBUG",
    #         "propagate": False,
    #     },
    # },

    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        },
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["default"],
    },
}
