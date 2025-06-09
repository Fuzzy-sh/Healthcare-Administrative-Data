# setup_env.py
import os
import sys


def set_project_root(relative_to=__file__, levels_up=2):
    root = os.path.abspath(
        os.path.join(os.path.dirname(relative_to), *([".."] * levels_up))
    )
    if root not in sys.path:
        sys.path.insert(0, root)
