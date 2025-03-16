import logging
import lotus.dtype_extensions
import lotus.models
import lotus.vector_store
import lotus.nl_expression
import lotus.templates
import lotus.utils
from lotus.sem_ops import (
    load_sem_index,
    sem_agg,
    sem_extract,
    sem_filter,
    sem_index,
    sem_join,
    sem_map,
    sem_partition_by,
    sem_search,
    sem_sim_join,
    sem_cluster_by,
    sem_dedup,
    sem_topk,
)
import sys
from lotus.web_search import web_search, WebSearchCorpus
from lotus.settings import settings  # type: ignore[attr-defined]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout
)

# TODO(v): In the future, consider loguru, but requires a wider refactor.
class ColorFormatter(logging.Formatter):
    # Not all are used; 
    FORMATS = {
        logging.DEBUG: "\033[36m%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\033[0m",
        logging.INFO: "\033[32m%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\033[0m",
        logging.WARNING: "\033[33m%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\033[0m",
        logging.ERROR: "\033[31m%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\033[0m",
        # Critical is unused... but kept here for completeness.
        logging.CRITICAL: "\033[1;31m%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s\033[0m"
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

for handler in logging.root.handlers:
    handler.setFormatter(ColorFormatter())

logger = logging.getLogger(__name__)

__all__ = [
    "sem_map",
    "sem_filter",
    "sem_agg",
    "sem_extract",
    "sem_join",
    "sem_partition_by",
    "sem_topk",
    "sem_index",
    "load_sem_index",
    "sem_sim_join",
    "sem_cluster_by",
    "sem_search",
    "sem_dedup",
    "settings",
    "nl_expression",
    "templates",
    "logger",
    "models",
    "vector_store",
    "utils",
    "dtype_extensions",
    "web_search",
    "WebSearchCorpus",
]
