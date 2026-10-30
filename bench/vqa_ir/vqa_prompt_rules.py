"""The VQA prompt used for every model in the paper: preamble, rules, answer format.

build_prompt() assembles, in order: a preamble (single image, multi-frame, or
the navigation context-tour preamble), the answer-format instruction, the rule
blocks that apply to the question (spatial rules for direction questions,
camera-distance rules for camera-distance questions, grid/layer counting rules
for locate questions), then the question and its options. vqa_prompts.py
picks the arguments for a released question record.
"""

import re

# Default random seed for model inference reproducibility
import os

DEFAULT_SEED = 42

PRE_PROMPT = """Answer the question based on the given image(s)."""

# Multi-frame pre-prompts
MULTI_FRAME_PRE_PROMPT = """These are consecutive observations from the head camera of a robot. Answer the question based on the given images."""

MULTI_FRAME_ROTATION_CONTEXT = """The robot is rotating in place. The images show different viewpoints as the robot turns."""

MULTI_FRAME_NAVIGATION_CONTEXT = """The robot is navigating through the environment. The images show the robot's observations as it moves."""

# Move-to + trajectory context (first frames = prior context, last = current view).
# Used when ``dataset_type == "move_to_context"`` in ``build_prompt``.
#
# NAV_PROMPT env var selects the variant (default "nav", used for the paper):
#   legacy : the original one-liner + the generic MULTI_FRAME_PRE_PROMPT + SPATIAL_RULES.
#            SPATIAL_RULES is the MANIPULATION definition (front/back = depth order
#            between two visible objects) — it contradicts the nav GT, where "back"
#            means BEHIND THE ROBOT, out of view. Kept only for comparison.
#   nav    : robot-centric definitions, states the target may be out of view, does
#            NOT claim the frames are consecutive, and skips SPATIAL_RULES. Images
#            stay unlabeled by design (the paper's prompt).
MULTI_FRAME_MOVE_TO_CONTEXT_LEGACY = (
    "The first few images are prior context along a movement trajectory in the environment."
    "The last image is the robot's current camera view. "
)
MULTI_FRAME_MOVE_TO_CONTEXT_NAV = """You are looking through the head camera of a mobile robot.

All images except the last one are a CONTEXT TOUR: earlier views recorded while moving through this environment. Use them to learn where things are. They were NOT taken from the robot's current position.

The LAST image is the robot's CURRENT camera view. All answers are relative to the robot's current position and heading in that last image.

Definitions (robot-centric, top-down):
- FRONT = the direction the camera is facing; BACK = directly behind the robot.
- LEFT / RIGHT = the robot's left and right when facing forward.
- The target may be OUT OF VIEW (behind the robot, or behind a wall). It still has a direction. Locate the target from the tour, locate the robot from the current view, and answer with the quadrant the target lies in."""

NAV_PROMPT_VARIANT = os.environ.get("NAV_PROMPT", "nav")
MULTI_FRAME_MOVE_TO_CONTEXT = (MULTI_FRAME_MOVE_TO_CONTEXT_LEGACY if NAV_PROMPT_VARIANT == "legacy"
                               else MULTI_FRAME_MOVE_TO_CONTEXT_NAV)

PRE_PROMPT_MCQ = """
Answer with the option's letter and text from the given choices directly.
Example: if you think the correct answer is 'C.front' from 'A.above B.under C. front D.behind.', your response should only be 'C.front'.
"""

PRE_PROMPT_NA = """
Do not respond with anything other than a single number!
"""

SPATIAL_RULES = """
Spatial Understanding Rules:
- Use the camera's perspective as the reference frame for all spatial relationships (unless explicitly stated otherwise).
- Front/Back: The object that appears farther from the camera (deeper into the scene) is considered "in front of" the object that appears closer to the camera. The object closer to the camera is at the "back" of the farther object.
- Left/Right: Defined from the camera's viewpoint. An object located on the left side of the image relative to another object is considered "left of" that object, and similarly for right.
- Above/Below: Determined by relative vertical position in the scene.
"""

LOCATE_RULES = """
Locate Question Rules:
- Grid Positions (Row x Column): Rows are counted from bottom to top. Columns are counted from left to right. For example, "1x1" is the bottom-left grid, "2x3" is the second row from the bottom and third column from the left.
- Layers: Counted from bottom to top of the container. Layer 1 is the bottommost level (closest to the floor/base), and higher layers are numbered sequentially upward.
- Levels: Counted from bottom to top, described by their relative height within the container. (e.g., "low level," "middle level," "high level," or specific numerical levels)
- Please ONLY consider the internal space and organizational structure of the specific container mentioned in the question.
"""

CAMERA_DISTANCE_RULES = """
Camera Distance Question Rules:
- When questions refer to "the camera", note that the robot hand visible in the image is NOT the camera. The camera is the viewpoint from which the image was captured, which is separate from any robot hand or manipulator visible in the scene.
- Distance is defined along the view trace from the camera to the 3Dcenter of the object.
"""

def load_system_prompt() -> str:
    """
    Get the full system prompt (PRE_PROMPT) for backward compatibility.
    Returns:
        str: The pre-prompt text
    """
    return PRE_PROMPT

def is_locate_question(question: str, options: str = None) -> bool:
    """
    Detect if this is a locate question (asking where an object is located on a container).
    
    Args:
        question: The question text
        options: Optional multiple-choice options (can be str, dict, or None)
    
    Returns:
        bool: True if this appears to be a locate question
    """
    # Check if question contains "located on" or similar patterns
    question_lower = question.lower()
    if "located on" in question_lower or "location on" in question_lower:
        return True
    
    # Check if options contain grid patterns (e.g., "1x1", "2x3") or layer/level patterns
    if options:
        # Convert options to string if needed
        if isinstance(options, dict):
            if 'options' in options:
                options_str = str(options['options'])
            else:
                options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
        else:
            options_str = str(options)
        
        options_lower = options_str.lower()
        # Check for grid patterns (e.g., "1x1", "2x3", "3x4")
        if re.search(r'\d+x\d+', options_lower):
            return True
        # Check for layer/level patterns
        if "layer" in options_lower or "level" in options_lower:
            return True
    
    return False


def is_direction_question(question: str, options: str = None) -> bool:
    """
    Detect if this is a direction question (asking about spatial relationships between objects).
    
    Args:
        question: The question text
        options: Optional multiple-choice options (can be str, dict, or None)
    
    Returns:
        bool: True if this appears to be a direction question (but not a locate question)
    """
    # First check if it's a locate question - if so, it's not a direction question
    if is_locate_question(question, options):
        return False
    
    # Convert options to string if needed
    options_str = None
    if options:
        if isinstance(options, dict):
            if 'options' in options:
                options_str = str(options['options'])
            else:
                options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
        else:
            options_str = str(options)
    
    # Check if question contains direction-related keywords
    question_lower = question.lower()
    direction_keywords = ["relative to", "with respect to", "direction"]
    if any(keyword in question_lower for keyword in direction_keywords):
        # Check if options contain spatial direction words
        if options_str:
            options_lower = options_str.lower()
            spatial_directions = ["front", "back", "behind", "left", "right", "above", "below", "under", "over"]
            if any(direction in options_lower for direction in spatial_directions):
                return True
    
    return False


def is_distance_question(question: str, options: str = None) -> bool:
    """
    Detect if this is a distance question (asking about distances between objects).
    
    Args:
        question: The question text
        options: Optional multiple-choice options (can be str, dict, or None)
    
    Returns:
        bool: True if this appears to be a distance question
    """
    question_lower = question.lower()
    distance_keywords = ["distance", "how far", "closer", "farther", "further", "closest", "farthest", "furthest"]
    if any(keyword in question_lower for keyword in distance_keywords):
        return True
    
    return False


def is_camera_question(question: str, options: str = None) -> bool:
    """
    Detect if this is a camera-related question (asking about spatial relationships with respect to the camera).
    
    Args:
        question: The question text
        options: Optional multiple-choice options (can be str, dict, or None)
    
    Returns:
        bool: True if this appears to be a camera-related question
    """
    question_lower = question.lower()
    camera_keywords = ["camera", "from the camera", "to the camera", "camera's"]
    if any(keyword in question_lower for keyword in camera_keywords):
        return True
    
    # Check options too
    if options:
        if isinstance(options, dict):
            if 'options' in options:
                options_str = str(options['options'])
            else:
                options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
        else:
            options_str = str(options)
        
        options_lower = options_str.lower()
        if any(keyword in options_lower for keyword in camera_keywords):
            return True
    
    return False


def build_prompt(
    question: str, 
    options: str = None, 
    is_numerical: bool = False,
    question_category: str = None,
    question_type: str = None,
    is_multi_frame: bool = False,
    dataset_type: str = None,
) -> str:
    """
    Build the complete prompt: PRE_PROMPT + question + POST_PROMPT
    
    Args:
        question: The question text
        options: Optional multiple-choice options (can be str, dict, or None)
        is_numerical: Whether this is a numerical answer question
        question_category: Optional question category (e.g., "direction", "distance", "size").
                          If provided, used directly instead of detection.
        question_type: Optional question type (e.g., "locate", "object_object", "camera").
                       If provided, used directly instead of detection.
        is_multi_frame: Whether this is a multi-frame question.
        dataset_type: Optional dataset type for multi-frame questions
            ("rotation", "navigation", or "move_to_context").
    
    Returns:
        str: Complete formatted prompt
    """
    # Convert options to string if it's not already
    options_str = None
    if options is not None:
        if isinstance(options, dict):
            # If options is a dict, try to extract or convert it
            # Common patterns: {'A': 'value', 'B': 'value'} or {'options': 'A: value, B: value'}
            if 'options' in options:
                options_str = str(options['options'])
            else:
                # Convert dict to string format: "A: value, B: value"
                options_str = ", ".join([f"{k}: {v}" for k, v in options.items()])
        elif isinstance(options, str):
            options_str = options
        else:
            options_str = str(options)
    
    # Choose the appropriate pre-prompt based on single vs multi-frame
    _nav = (is_multi_frame and dataset_type == "move_to_context" and NAV_PROMPT_VARIANT != "legacy")
    if _nav:
        # nav prompt is self-contained: no "consecutive observations" preamble
        prompt = MULTI_FRAME_MOVE_TO_CONTEXT_NAV + "\n\n"
    elif is_multi_frame:
        prompt = MULTI_FRAME_PRE_PROMPT + "\n\n"
        # Add rotation/navigation-specific context
        if dataset_type == "rotation":
            prompt += MULTI_FRAME_ROTATION_CONTEXT + "\n\n"
        elif dataset_type == "navigation":
            prompt += MULTI_FRAME_NAVIGATION_CONTEXT + "\n\n"
        elif dataset_type == "move_to_context":
            prompt += MULTI_FRAME_MOVE_TO_CONTEXT + "\n\n"
    else:
        prompt = PRE_PROMPT + "\n\n"
    
    if is_numerical:
        prompt += PRE_PROMPT_NA
    elif options_str:
        prompt += PRE_PROMPT_MCQ
    
    # Determine if this is a locate question (use provided type or detect)
    is_locate = False
    if question_type:
        is_locate = (question_type.lower() == "locate")
    else:
        is_locate = is_locate_question(question, options_str)
    
    # Determine if this is a direction question (use provided category/type or detect)
    is_direction = False
    if question_category:
        is_direction = (question_category.lower() == "direction" and not is_locate)
    else:
        is_direction = is_direction_question(question, options_str)
    
    # Determine if this is a distance question (use provided category or detect)
    is_distance = False
    if question_category:
        is_distance = (question_category.lower() == "distance")
    else:
        is_distance = is_distance_question(question, options_str)
    
    # Determine if this is a camera-related question (use provided type or detect)
    is_camera = False
    if question_type:
        is_camera = (question_type.lower() == "camera")
    else:
        is_camera = is_camera_question(question, options_str)
    
    # Add spatial rules if this is a direction question but NOT a locate question
    if is_direction and not _nav:
        # SPATIAL_RULES is the manipulation (object-vs-object) definition — never
        # attach it to the robot-centric nav questions.
        prompt += "\n\n" + SPATIAL_RULES
    
    # Add camera distance rules only if this is a distance question AND a camera question
    # (Camera reference and depth-based distance are only relevant for camera-based distance questions)
    if is_distance and is_camera:
        prompt += "\n\n" + CAMERA_DISTANCE_RULES
    
    # Add locate-specific rules if this is a locate question
    if is_locate:
        prompt += "\n\n" + LOCATE_RULES
    
    prompt += "\n\n" + "Question: " + question
    
    if options_str:
        prompt += "\n\n" + "Options: " + options_str
    
    return prompt

