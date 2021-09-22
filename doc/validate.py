import sys

from numpydoc.validate import validate

import alphacsc


def validate_doc():
    # dir(alphacsc), exclude patterns
    to_validate = [obj for obj in ["alphacsc.learn_d_z_multi.learn_d_z_multi"] ]
    obj_errors = [validate(name) for name in to_validate]
    for obj_error in obj_errors:
        for error in obj_error['errors']:
            print(f"{obj_error['file']}:{obj_error['file_line']} - {error}")
    return obj_errors

if __name__ == "__main__":
    obj_errors = validate_doc()
    sys.exit(0 if not obj_errors else 1)