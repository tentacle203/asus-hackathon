# config.py

# =========================
# Path
# =========================

DATABASE_FOLDER = "train_data/train_images"
METADATA_PATH = "train_data/metadata.json"

QUERY_FOLDER = "test_data/test_images"
OUTPUT_FOLDER = "outputs"


# =========================
# DINOv2
# =========================

DINO_MODEL = "facebook/dinov2-base"


# =========================
# Coarse Retrieval
# =========================

# 第一階段：
# 80 → 20
GLOBAL_TOP_K = 20

# 第二階段：
# 20 → 5
LOCAL_TOP_K = 5

# Global / Local weighting
GLOBAL_WEIGHT = 0.4
LOCAL_WEIGHT = 0.6

# Local patch matching
LOCAL_TOP_RATIO = 0.20


# =========================
# Agent 1
# =========================

CONF_THRESHOLD = 0.35


# =========================
# Direction
# =========================

QUERY_DIRECTIONS = [
    "east",
    "west",
    "south",
    "north"
]


# =========================
# Agent 3
# =========================

# Agent 3 最後保留多少個 GPS candidate
GPS_TOP_K = 10

# 是否考慮照片拍攝方向
USE_HEADING_SCORE = True

# GPS candidate score 中
# visual similarity 的權重
GPS_VISUAL_WEIGHT = 0.85

# heading compatibility 的權重
GPS_HEADING_WEIGHT = 0.15