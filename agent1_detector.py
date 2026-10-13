import os
import cv2
import math
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ==========================================================
# Confidence Threshold
# ==========================================================

CONF_THRESHOLD = 0.35

MODEL_NAME = os.path.expanduser(
    "~/.cache/huggingface/hub/models--microsoft--Florence-2-base/"
    "snapshots/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac"
)


# ==========================================================
# Agent 1
# ==========================================================

class Agent1Detector:

    def __init__(
        self,
        model_name=MODEL_NAME,
        custom_classes=None,
        device=None,
        enable_tiled=True,
        grid_size=3,
        tile_overlap=0.15
    ):

        # ==================================================
        # Device
        # ==================================================

        if device is None:

            if torch.backends.mps.is_available():
                device = "mps"

            elif torch.cuda.is_available():
                device = "cuda"

            else:
                device = "cpu"

        self.device = device

        print(
            f"🤖 [Agent 1] Device: "
            f"{self.device}"
        )

        print(
            "🤖 [Agent 1] 載入 "
            "Florence-2 模型..."
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                model_name,
                trust_remote_code=True
            )
            .to(self.device)
            .eval()
        )

        self.processor = (
            AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                local_files_only=True
            )
        )


        print(
            "✅ [Agent 1] "
            "Florence-2 載入完成"
        )

        # ==================================================
        # Detection Classes
        #
        # 注意：
        # 不再使用 apartment
        # 改成比較一般的 building
        # ==================================================

        if custom_classes is None:

            self.custom_classes = [

                # ------------------------------
                # 建築 / 環境
                # ------------------------------

                # "building",
                "window",
                "door"
                "pathway",
                # "tree",
                "trimmed bush",
                "fence",

                # ------------------------------
                # 公園設施
                # ------------------------------

                "bench",
                "playground equipment",

                # ------------------------------
                # 垃圾 / 標示
                # ------------------------------

                "trash can",
                "street sign",

                # ------------------------------
                # 路邊設施
                # ------------------------------

                "street lamp",
                "railing",
                "metal railing",

                # ------------------------------
                # 運動設施
                # ------------------------------

                "exercise equipment",
                "pull up bar",

                # ------------------------------
                # 樹木細部
                # ------------------------------

                "tree trunk",
                "tree base",
                "tree branch",

                # ------------------------------
                # 其他
                # ------------------------------

                "playground tire",
            ]

        else:

            self.custom_classes = custom_classes

        # ==================================================
        # Tiled Detection
        #
        # 不需要所有 class 都切 3x3。
        #
        # 主要針對：
        #   building
        #   street lamp
        #   street sign
        #   bench
        #   playground equipment
        #   exercise equipment
        #   pull up bar
        #   railing
        #
        # 這些物件比較可能出現在遠方。
        # ==================================================

        self.enable_tiled = enable_tiled

        self.grid_size = grid_size

        self.tile_overlap = tile_overlap

        self.tile_classes = [

            "building",
            "street lamp",
            "street sign",
            "bench",
            "playground equipment",
            "exercise equipment",
            "pull up bar",
            "railing",
            "metal railing",

        ]

        print(
            f"✅ [Agent 1] 載入 "
            f"{len(self.custom_classes)} "
            f"種偵測標籤"
        )

        if self.enable_tiled:

            print(
                f"🔍 [Agent 1] "
                f"Tiled Detection: "
                f"{self.grid_size}x{self.grid_size}"
            )

            print(
                f"🔍 [Agent 1] "
                f"Tile overlap: "
                f"{self.tile_overlap:.0%}"
            )

            print(
                "🔎 [Agent 1] "
                "遠方物件 Tile classes:"
            )

            for target in self.tile_classes:
                print(f"   - {target}")

    # ======================================================
    # Confidence
    # ======================================================

    @staticmethod
    def _score_to_confidence(score):

        """
        將 Florence-2 score
        轉成 0~1。

        注意：
        這不是 YOLO probability。

        主要用於 threshold filtering。
        """

        if score is None:
            return None

        try:

            score = float(score)

            confidence = math.exp(score)

            confidence = max(
                0.0,
                min(
                    1.0,
                    confidence
                )
            )

            return confidence

        except Exception:

            return None

    # ======================================================
    # Florence-2 Single Object Detection
    # ======================================================

    def _detect_single_object(
        self,
        image,
        target
    ):

        task_prompt = (
            "<OPEN_VOCABULARY_DETECTION>"
        )

        prompt = (
            task_prompt +
            target
        )

        # --------------------------------------------------
        # Processor
        # --------------------------------------------------

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if hasattr(v, "to")
        }

        # --------------------------------------------------
        # Florence-2
        # --------------------------------------------------

        with torch.no_grad():

            generated = (
                self.model.generate(

                    input_ids=inputs[
                        "input_ids"
                    ],

                    pixel_values=inputs[
                        "pixel_values"
                    ],

                    max_new_tokens=1024,

                    num_beams=3,

                    do_sample=False,

                    return_dict_in_generate=True,

                    output_scores=True
                )
            )

        # --------------------------------------------------
        # Decode
        # --------------------------------------------------

        generated_text = (
            self.processor.batch_decode(
                generated.sequences,
                skip_special_tokens=False
            )[0]
        )

        # --------------------------------------------------
        # Transition Score
        # --------------------------------------------------

        try:

            transition_scores = (
                self.model.compute_transition_scores(
                    sequences=generated.sequences,
                    scores=generated.scores,
                    beam_indices=generated.beam_indices,
                    normalize_logits=True
                )
            )

        except TypeError:

            transition_scores = (
                self.model.compute_transition_scores(
                    sequences=generated.sequences,
                    scores=generated.scores,
                    beam_indices=generated.beam_indices
                )
            )

        # --------------------------------------------------
        # Post Processing
        # --------------------------------------------------

        try:

            results = (
                self.processor
                .post_process_generation(

                    sequence=generated.sequences[0],

                    transition_beam_score=(
                        transition_scores[0]
                    ),

                    task=task_prompt,

                    image_size=image.size
                )
            )

        except TypeError:

            results = (
                self.processor
                .post_process_generation(

                    generated_text,

                    task=task_prompt,

                    image_size=image.size
                )
            )

            transition_scores = None

        # --------------------------------------------------
        # Extract result
        # --------------------------------------------------

        det_result = results.get(
            task_prompt,
            {}
        )

        bboxes = det_result.get(
            "bboxes",
            []
        )

        labels = det_result.get(
            "bboxes_labels",
            []
        )

        if not labels:

            labels = det_result.get(
                "labels",
                []
            )

        scores = det_result.get(
            "scores",
            None
        )

        if scores is None:

            scores = det_result.get(
                "confidence",
                None
            )

        detections = []

        # --------------------------------------------------
        # Convert detections
        # --------------------------------------------------

        for i, box in enumerate(
            bboxes
        ):

            if i < len(labels):

                label = labels[i]

            else:

                label = target

            confidence = None

            # --------------------------------------------------
            # Florence output score
            # --------------------------------------------------

            if (
                scores is not None
                and
                i < len(scores)
            ):

                try:

                    confidence = float(
                        scores[i]
                    )

                    if (
                        confidence < 0
                        or
                        confidence > 1
                    ):

                        confidence = (
                            self._score_to_confidence(
                                confidence
                            )
                        )

                except Exception:

                    confidence = None

            # --------------------------------------------------
            # Transition score fallback
            # --------------------------------------------------

            elif (
                transition_scores
                is not None
            ):

                try:

                    raw_score = (
                        transition_scores[0]
                        .mean()
                        .item()
                    )

                    confidence = (
                        self._score_to_confidence(
                            raw_score
                        )
                    )

                except Exception:

                    confidence = None

            detections.append({

                "label": label,

                "box": box,

                "confidence": confidence

            })

        return detections

    # ======================================================
    # Confidence Filter
    # ======================================================

    @staticmethod
    def _filter_by_confidence(
        detections,
        conf_threshold
    ):

        filtered = []

        for detection in detections:

            confidence = (
                detection.get(
                    "confidence"
                )
            )

            if confidence is None:

                filtered.append(
                    detection
                )

            elif (
                confidence >=
                conf_threshold
            ):

                filtered.append(
                    detection
                )

        return filtered

    # ======================================================
    # IoU
    # ======================================================

    @staticmethod
    def _calculate_iou(
        box1,
        box2
    ):

        x1 = max(
            box1[0],
            box2[0]
        )

        y1 = max(
            box1[1],
            box2[1]
        )

        x2 = min(
            box1[2],
            box2[2]
        )

        y2 = min(
            box1[3],
            box2[3]
        )

        intersection_w = max(
            0,
            x2 - x1
        )

        intersection_h = max(
            0,
            y2 - y1
        )

        intersection = (
            intersection_w *
            intersection_h
        )

        area1 = (
            max(
                0,
                box1[2] - box1[0]
            )
            *
            max(
                0,
                box1[3] - box1[1]
            )
        )

        area2 = (
            max(
                0,
                box2[2] - box2[0]
            )
            *
            max(
                0,
                box2[3] - box2[1]
            )
        )

        union = (
            area1 +
            area2 -
            intersection
        )

        if union <= 0:

            return 0.0

        return (
            intersection /
            union
        )

    # ======================================================
    # NMS
    # ======================================================

    def _nms_by_label(
        self,
        detections,
        iou_threshold=0.5
    ):

        """
        對相同 label 的重複 detection
        做 NMS。

        主要解決：

        原圖
          ↓
        Tile 1
          ↓
        Tile 2
          ↓
        同一個 building 被抓兩次

        最後只保留較好的那一個。
        """

        if not detections:

            return []

        grouped = {}

        # --------------------------------------------------
        # Group by label
        # --------------------------------------------------

        for detection in detections:

            label = detection.get(
                "label",
                "unknown"
            )

            if label not in grouped:

                grouped[label] = []

            grouped[label].append(
                detection
            )

        final_detections = []

        # --------------------------------------------------
        # NMS each label
        # --------------------------------------------------

        for label, items in grouped.items():

            # confidence 高 → 優先
            items = sorted(
                items,
                key=lambda x:
                    x.get(
                        "confidence"
                    )
                    if x.get(
                        "confidence"
                    ) is not None
                    else 0.0,
                reverse=True
            )

            kept = []

            while items:

                current = items.pop(0)

                kept.append(
                    current
                )

                remaining = []

                for item in items:

                    iou = (
                        self._calculate_iou(
                            current["box"],
                            item["box"]
                        )
                    )

                    if iou < iou_threshold:

                        remaining.append(
                            item
                        )

                items = remaining

            final_detections.extend(
                kept
            )

        return final_detections

    # ======================================================
    # Create 3x3 Tiles
    # ======================================================

    def _create_tiles(
        self,
        image
    ):

        """
        將原始圖片切成 3x3。

        overlap 可以避免物件剛好落在
        tile 邊界時被切掉。

        回傳：

        [
            {
                "image": tile_image,
                "offset_x": ...,
                "offset_y": ...,
                "tile_id": ...
            }
        ]
        """

        width, height = image.size

        rows = self.grid_size
        cols = self.grid_size

        # --------------------------------------------------
        # 計算 tile size
        # --------------------------------------------------

        base_tile_w = width / cols
        base_tile_h = height / rows

        tile_w = int(
            base_tile_w *
            (1.0 + self.tile_overlap)
        )

        tile_h = int(
            base_tile_h *
            (1.0 + self.tile_overlap)
        )

        tiles = []

        # --------------------------------------------------
        # Generate tiles
        # --------------------------------------------------

        for row in range(rows):

            for col in range(cols):

                # 原始 grid 起點
                center_x = (
                    col + 0.5
                ) * base_tile_w

                center_y = (
                    row + 0.5
                ) * base_tile_h

                x1 = int(
                    center_x -
                    tile_w / 2
                )

                y1 = int(
                    center_y -
                    tile_h / 2
                )

                x2 = x1 + tile_w

                y2 = y1 + tile_h

                # --------------------------------------------------
                # Clamp
                # --------------------------------------------------

                if x1 < 0:

                    x1 = 0

                if y1 < 0:

                    y1 = 0

                if x2 > width:

                    x2 = width

                if y2 > height:

                    y2 = height

                # --------------------------------------------------
                # Crop
                # --------------------------------------------------

                tile_image = (
                    image.crop(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        )
                    )
                )

                tiles.append({

                    "image":
                        tile_image,

                    "offset_x":
                        x1,

                    "offset_y":
                        y1,

                    "tile_id":
                        f"{row}_{col}",

                    "box":
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ]
                })

        return tiles

    # ======================================================
    # Convert Tile BBox → Original Image BBox
    # ======================================================

    @staticmethod
    def _tile_box_to_original(
        box,
        offset_x,
        offset_y
    ):

        x1, y1, x2, y2 = map(
            float,
            box
        )

        x1 += offset_x
        x2 += offset_x

        y1 += offset_y
        y2 += offset_y

        return [
            int(x1),
            int(y1),
            int(x2),
            int(y2)
        ]

    # ======================================================
    # Main Detection
    # ======================================================

    def detect_and_crop(
        self,
        image_path,
        output_dir="agent1_output",
        conf_threshold=CONF_THRESHOLD,
        imgsz=1024
    ):

        # ==================================================
        # Check image
        # ==================================================

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"❌ 找不到圖片："
                f"{image_path}"
            )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        img_name = (
            os.path.splitext(
                os.path.basename(
                    image_path
                )
            )[0]
        )

        # ==================================================
        # OpenCV image
        # ==================================================

        frame = cv2.imread(
            image_path
        )

        if frame is None:

            raise ValueError(
                f"❌ OpenCV 無法讀取："
                f"{image_path}"
            )

        h, w, _ = frame.shape

        # ==================================================
        # PIL image
        # ==================================================

        image = (
            Image.open(
                image_path
            ).convert("RGB")
        )

        print(
            "\n========================================"
        )

        print(
            "🔍 [Agent 1] "
            "開始掃描圖片"
        )

        print(
            f"📐 Image size: "
            f"{w} x {h}"
        )

        print(
            f"🎯 Confidence Threshold: "
            f"{conf_threshold}"
        )

        print(
            f"🔲 Tiled Detection: "
            f"{self.enable_tiled}"
        )

        print(
            "========================================"
        )

        # ==================================================
        # All detections
        # ==================================================

        all_detections = []

        # ==================================================
        # PART 1
        # Original Full Image Detection
        # ==================================================

        print(
            "\n"
            "========================================"
        )

        print(
            "🖼️ PART 1 — 原圖偵測"
        )

        print(
            "========================================"
        )

        for idx, target in enumerate(
            self.custom_classes
        ):

            print(
                f"\n[{idx + 1}/"
                f"{len(self.custom_classes)}]"
                f" 🔎 Searching: "
                f"{target}"
            )

            try:

                detections = (
                    self._detect_single_object(
                        image,
                        target
                    )
                )

                print(
                    f"    Florence 找到："
                    f"{len(detections)}"
                )

                filtered = (
                    self._filter_by_confidence(
                        detections,
                        conf_threshold
                    )
                )

                print(
                    f"    Threshold 過濾後："
                    f"{len(filtered)}"
                )

                # --------------------------------------------------
                # Save original detections
                # --------------------------------------------------

                for detection in filtered:

                    box = detection[
                        "box"
                    ]

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    # Clamp

                    x1 = max(
                        0,
                        min(x1, w)
                    )

                    y1 = max(
                        0,
                        min(y1, h)
                    )

                    x2 = max(
                        0,
                        min(x2, w)
                    )

                    y2 = max(
                        0,
                        min(y2, h)
                    )

                    if (
                        x2 <= x1
                        or
                        y2 <= y1
                    ):

                        continue

                    confidence = (
                        detection.get(
                            "confidence"
                        )
                    )

                    all_detections.append({

                        "label":
                            target,

                        "detected_label":
                            detection[
                                "label"
                            ],

                        "box": [
                            x1,
                            y1,
                            x2,
                            y2
                        ],

                        "confidence":
                            confidence,

                        "source":
                            "full_image",

                        "tile_id":
                            None
                    })

                    if confidence is None:

                        confidence_text = "N/A"

                    else:

                        confidence_text = (
                            f"{confidence:.3f}"
                        )

                    print(
                        f"       ✅ "
                        f"{target}"
                        f" | confidence="
                        f"{confidence_text}"
                        f" | box="
                        f"{[x1,y1,x2,y2]}"
                    )

            except Exception as e:

                print(
                    f"    ⚠️ "
                    f"{target} 偵測失敗："
                    f"{e}"
                )

        # ==================================================
        # PART 2
        # 3x3 Tiled Detection
        # ==================================================

        if self.enable_tiled:

            print(
                "\n"
                "========================================"
            )

            print(
                "🔲 PART 2 — 3×3 遠方物件偵測"
            )

            print(
                "========================================"
            )

            tiles = self._create_tiles(
                image
            )

            print(
                f"📦 建立 "
                f"{len(tiles)} "
                f"個 tiles"
            )

            # --------------------------------------------------
            # Only run important classes
            # --------------------------------------------------

            for target in self.tile_classes:

                if target not in self.custom_classes:

                    continue

                print(
                    "\n"
                    f"🔎 Tiled Searching: "
                    f"{target}"
                )

                for tile in tiles:

                    tile_image = (
                        tile["image"]
                    )

                    tile_id = (
                        tile["tile_id"]
                    )

                    offset_x = (
                        tile["offset_x"]
                    )

                    offset_y = (
                        tile["offset_y"]
                    )

                    print(
                        f"   Tile {tile_id}"
                    )

                    try:

                        detections = (
                            self._detect_single_object(
                                tile_image,
                                target
                            )
                        )

                        filtered = (
                            self._filter_by_confidence(
                                detections,
                                conf_threshold
                            )
                        )

                        if not filtered:

                            continue

                        for detection in filtered:

                            # --------------------------------------------------
                            # Tile bbox
                            # --------------------------------------------------

                            tile_box = (
                                detection["box"]
                            )

                            # --------------------------------------------------
                            # Convert to original image
                            # --------------------------------------------------

                            original_box = (
                                self._tile_box_to_original(
                                    tile_box,
                                    offset_x,
                                    offset_y
                                )
                            )

                            x1, y1, x2, y2 = (
                                original_box
                            )

                            # --------------------------------------------------
                            # Clamp
                            # --------------------------------------------------

                            x1 = max(
                                0,
                                min(x1, w)
                            )

                            y1 = max(
                                0,
                                min(y1, h)
                            )

                            x2 = max(
                                0,
                                min(x2, w)
                            )

                            y2 = max(
                                0,
                                min(y2, h)
                            )

                            if (
                                x2 <= x1
                                or
                                y2 <= y1
                            ):

                                continue

                            confidence = (
                                detection.get(
                                    "confidence"
                                )
                            )

                            all_detections.append({

                                "label":
                                    target,

                                "detected_label":
                                    detection[
                                        "label"
                                    ],

                                "box": [
                                    x1,
                                    y1,
                                    x2,
                                    y2
                                ],

                                "confidence":
                                    confidence,

                                "source":
                                    "tile",

                                "tile_id":
                                    tile_id
                            })

                            if confidence is None:

                                confidence_text = "N/A"

                            else:

                                confidence_text = (
                                    f"{confidence:.3f}"
                                )

                            print(
                                f"      ✅ "
                                f"{target}"
                                f" | confidence="
                                f"{confidence_text}"
                                f" | original box="
                                f"{[x1,y1,x2,y2]}"
                            )

                    except Exception as e:

                        print(
                            f"      ⚠️ "
                            f"Tile {tile_id} "
                            f"偵測失敗："
                            f"{e}"
                        )

        # ==================================================
        # PART 3
        # NMS
        # ==================================================

        print(
            "\n"
            "========================================"
        )

        print(
            "🧹 PART 3 — 移除重複偵測"
        )

        print(
            "========================================"
        )

        before_nms = len(
            all_detections
        )

        all_detections = (
            self._nms_by_label(
                all_detections,
                iou_threshold=0.5
            )
        )

        after_nms = len(
            all_detections
        )

        print(
            f"📦 NMS 前："
            f"{before_nms}"
        )

        print(
            f"📦 NMS 後："
            f"{after_nms}"
        )

        # ==================================================
        # PART 4
        # Crop + Draw
        # ==================================================

        print(
            "\n"
            "========================================"
        )

        print(
            "✂️ PART 4 — Crop + Draw"
        )

        print(
            "========================================"
        )

        cropped_objects = []

        annotated_frame = (
            frame.copy()
        )

        # --------------------------------------------------
        # Process final detections
        # --------------------------------------------------

        for idx, detection in enumerate(
            all_detections
        ):

            label = detection[
                "label"
            ]

            detected_label = (
                detection[
                    "detected_label"
                ]
            )

            confidence = (
                detection[
                    "confidence"
                ]
            )

            x1, y1, x2, y2 = (
                detection["box"]
            )

            # --------------------------------------------------
            # Crop
            # --------------------------------------------------

            cropped_img = frame[
                y1:y2,
                x1:x2
            ]

            if cropped_img.size == 0:

                continue

            # --------------------------------------------------
            # Filename
            # --------------------------------------------------

            safe_label = (
                label
                .replace(
                    " ",
                    "_"
                )
                .replace(
                    "/",
                    "_"
                )
            )

            crop_save_path = os.path.join(

                output_dir,

                f"{img_name}_crop_"
                f"{idx}_{safe_label}.jpg"

            )

            # --------------------------------------------------
            # Save crop
            # --------------------------------------------------

            cv2.imwrite(
                crop_save_path,
                cropped_img
            )

            # --------------------------------------------------
            # Draw bbox
            # --------------------------------------------------

            cv2.rectangle(

                annotated_frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )

            # --------------------------------------------------
            # Label
            # --------------------------------------------------

            if confidence is None:

                text = (
                    f"{label}"
                )

            else:

                text = (
                    f"{label} "
                    f"{confidence:.2f}"
                )

            cv2.putText(

                annotated_frame,

                text,

                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.55,

                (0, 255, 0),

                2

            )

            # --------------------------------------------------
            # Agent 2 Data
            # --------------------------------------------------

            cropped_objects.append({

                "id":
                    idx,

                "label":
                    label,

                "detected_label":
                    detected_label,

                "confidence":
                    confidence,

                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "crop_img":
                    cropped_img,

                "crop_path":
                    crop_save_path,

                "source":
                    detection.get(
                        "source",
                        "unknown"
                    ),

                "tile_id":
                    detection.get(
                        "tile_id"
                    )
            })

        # ==================================================
        # Save annotated image
        # ==================================================

        annotated_img_path = os.path.join(

            output_dir,

            f"{img_name}_detected.jpg"

        )

        cv2.imwrite(

            annotated_img_path,

            annotated_frame

        )

        # ==================================================
        # Summary
        # ==================================================

        print(
            "\n"
            "========================================"
        )

        print(
            "✨ [Agent 1] "
            "Florence-2 偵測完成"
        )

        print(
            f"🎯 Confidence Threshold: "
            f"{conf_threshold}"
        )

        print(
            f"📦 最終保留物件："
            f"{len(cropped_objects)}"
        )

        print(
            f"🖼️ 標註圖片："
            f"{annotated_img_path}"
        )

        print(
            "========================================"
        )

        return (
            cropped_objects,
            annotated_img_path
        )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    # ======================================================
    # 建立 Agent 1
    # ======================================================

    agent1 = Agent1Detector(
        enable_tiled=True,
        grid_size=3,
        tile_overlap=0.15
    )

    test_image = (
        "test_images/node_21_03.jpg"
    )

    detected_items, output_image_path = (
        agent1.detect_and_crop(

            image_path=test_image,

            output_dir="agent1_output",

            conf_threshold=CONF_THRESHOLD,

            imgsz=1024

        )
    )

    # ======================================================
    # 印出最終結果
    # ======================================================

    print(
        "\n📋 Agent 1 最終偵測清單："
    )

    for item in detected_items:

        confidence = (
            item["confidence"]
        )

        if confidence is None:

            confidence_text = "N/A"

        else:

            confidence_text = (
                f"{confidence:.3f}"
            )

        print(

            f"  - 物件 #{item['id']}: "

            f"[{item['label']}] "

            f"| confidence="
            f"{confidence_text} "

            f"| BBox="
            f"{item['box']} "

            f"| source="
            f"{item.get('source')} "

            f"| tile="
            f"{item.get('tile_id')} "

            f"| Crop="
            f"{item['crop_path']}"

        )

