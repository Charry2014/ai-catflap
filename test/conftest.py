import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), '..', 'src')
for sub in ('catflap', 'modules'):
    sys.path.insert(0, os.path.abspath(os.path.join(ROOT, sub)))
sys.path.insert(0, os.path.abspath(os.path.join(ROOT)))     # so "tools" can be imported
