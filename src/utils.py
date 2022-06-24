import os
import sys

def mkdir_if_not_exist(inputdir):
    if not os.path.exists(inputdir):
        os.makedirs(inputdir)
    return inputdir

def eprint(args):
    sys.stderr.write(str(args) + "\n")