from __future__ import print_function

import logging

log = logging.getLogger('SQCommon')
log.addHandler(logging.StreamHandler())
if log.level == logging.NOTSET:
    log.setLevel(logging.INFO)


__all__ = ['Result', 'Stats']

from ._objective import ObjectiveFunction
from ._result import Result
from ._stats import Stats
