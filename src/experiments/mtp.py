"""MTP settings for the selected MLSpace worker."""

from .mtp_design import configuration
from .suite import SUITE

globals().update(vars(configuration(SUITE)))
