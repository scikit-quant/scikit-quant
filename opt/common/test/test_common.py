import py, os, sys
import numpy as np

sys.path = [os.path.join(os.pardir, 'python')] + sys.path


class TestCOMMON:
    def setup_class(cls):
        import SQCommon, logging
        SQCommon.log.setLevel(logging.DEBUG)

    def test01_API(self):
        """Test existence and semantics of the common API"""

        import SQCommon

        assert SQCommon.Result
        assert SQCommon.Stats
