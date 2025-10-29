import os
import sys

# Add ImageBind to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
imagebind_path = os.path.join(current_dir, '..', 'ImageBind')
if imagebind_path not in sys.path:
    sys.path.insert(0, imagebind_path)

from .videoragcontent import VideoRAG, QueryParam