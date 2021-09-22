import re
import sys

# XXX this only works with the current numpydoc master, 1.1.0 does not
# have get_doc_object
from numpydoc.docscrape import get_doc_object
from numpydoc.validate import validate

import alphacsc

EXCLUDE_PATTERNS = []

def is_excluded(name):
    for pat in EXCLUDE_PATTERNS:
        if re.search(pat, name) is not None:
            return True
    return False

def get_top_level_names():
    return {f"alphacsc.{name}" for name in dir(alphacsc) if not is_excluded(name)}


def validate_doc():
    obj_errors = [validate(doc_obj) for doc_obj in get_top_level_names()]
    for obj_error in obj_errors:
        for error in obj_error['errors']:
            print(f"{obj_error['file']}:{obj_error['file_line']} - {error}")
    return obj_errors

if __name__ == "__main__":
    obj_errors = validate_doc()
    sys.exit(0 if not obj_errors else 1)