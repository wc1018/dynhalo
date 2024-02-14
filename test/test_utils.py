import os

import pytest

from dhm import utils


def test_mkdir():
    dirpath_fail = '/zzz/tmp_dir_test/'
    with pytest.raises(FileNotFoundError):
        utils.mkdir(dirpath_fail, verbose=False)

    dirpath_pass = os.getcwd() + '/test/tmp_dir_test/'
    utils.mkdir(dirpath_pass, verbose=False)
    assert os.path.exists(dirpath_pass)
    os.removedirs(dirpath_pass)
