import os
from typing import List, Dict

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, 'workflow')

file_types = [
    ('Image', ('*.png', '*.jpg', '*.jpeg', '*.gif', '*.bmp')),
    ('Video', ('*.mp4', '*.mkv', '*.webm', '*.avi', '*.wmv')),
]

images_extensions = tuple([e.split(".")[1] for e in file_types[0][1]])
videos_extensions = tuple([e.split(".")[1] for e in file_types[1][1]])

enhancer_none = "None"
enhancer_best_face_only = "Best face only"
enhancer_faces_only = "Faces only"
enhancer_all = "All"
enhancer_options = [enhancer_none,
                    # enhancer_best_face_only,
                    enhancer_faces_only,
                    enhancer_all]
enhancer_option: str = enhancer_none

faces_best_one = "Best one"
faces_all = "All"
faces_none = "None"
faces_options = [faces_best_one,
                 faces_all,
                 faces_none]
face_option: str = faces_none

source_path = None
subject_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_frames = True
video_encoder = 'libx264'
video_quality = 18
max_memory = None
distance_score: int = 25
execution_providers: List[str] = []
execution_threads = None
log_level = 'info'
fp_ui: Dict[str, bool] = {}
nsfw = True
