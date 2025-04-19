#
# Copyright © 2024 QPerfect. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import logging.config


unhandled_greenlet_exception = False

def config_logger(loglevel: str, logfile: str = None):
    """
    Configure the logging settings for the application.
    
    Args:
        loglevel: The logging level as a string (e.g., 'DEBUG', 'INFO').
        logfile: Optional path to a file where logs should be saved.
    """
    loglevel = loglevel.upper()

    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": (
                    "%(log_color)s[%(asctime)s] %(levelname)s/%(name)s: "
                    "%(message)s"
                ),
                "log_colors": {
                    "DEBUG": "blue",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "bold_red,bg_white",
                },
                "secondary_log_colors": {},
                "style": "%",
            },
            "plain": {
                "format": "%(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "colorlog.StreamHandler",
                "formatter": "colored",
            },
            "console_plain": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
            },
        },
        "loggers": {
            "feniqs": {
                "handlers": ["console"],
                "level": loglevel,
               
                "propagate": False,
            },
            "feniqs.stats_logger": {
                "handlers": ["console_plain"],
                "level": "INFO",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": loglevel,
        },
    }

    if logfile:
        LOGGING_CONFIG["handlers"]["file"] = {
            "class": "logging.FileHandler",
            "filename": logfile,
            "formatter": "colored",
        }
        LOGGING_CONFIG["loggers"]["feniqs"]["handlers"].append("file")
        LOGGING_CONFIG["root"]["handlers"].append("file")

    logging.config.dictConfig(LOGGING_CONFIG)

def greenlet_exception_logger(logger: logging.Logger, level: int = logging.CRITICAL):
    """
    Create a handler function to log unhandled exceptions in greenlets.
    
    Args:
        logger: The logger to use for logging exceptions.
        level: The logging level for the exceptions.
    Returns:
        func: A function that logs exceptions from greenlets.
    """
    def exception_handler(greenlet):
        """
        Handle exceptions from a greenlet and log them appropriately.
        
        Args:
            greenlet: The greenlet that raised the exception.
        """
        if greenlet.exc_info[0] == SystemExit:
            logger.log(
                min(logging.INFO, level),  # Use INFO level for SystemExit exceptions.
                "sys.exit(%s) called (use log level DEBUG for callstack)" % greenlet.exc_info[1],
            )
            logger.log(logging.DEBUG, "Unhandled exception in greenlet: %s", greenlet, exc_info=greenlet.exc_info)
        else:
            logger.log(level, "Unhandled exception in greenlet: %s", greenlet, exc_info=greenlet.exc_info)
        
        global unhandled_greenlet_exception
        unhandled_greenlet_exception = True

    return exception_handler

