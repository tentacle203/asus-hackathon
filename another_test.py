
import os
import cv2
import math
import torch

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


# ==========================================================
# ⭐ Agent 1 設定區
# ==========================================================

# ==========================================================
# Confidence Threshold
#
# ⭐ 你主要調整這裡就可以
#
# 例如：
#
# 0.05 → 比較寬鬆
# 0.08 → 原本設定
# 0.10 → 中等
# 0.20 → 比較嚴格
# 0.50 → 非常嚴格
# ==========================================================

CONF_THRESHOLD = 0.2


# ==========================================================
# Florence-2 Model
# ==========================================================

MODEL_NAME = "microsoft/Florence-2-base"


# ==========================================================
# Agent 1
# ==========================================================

class Agent1Detector:
    """
    Agent 1:
    視覺物件自動偵測與擷取 Agent

    使用：
        Microsoft Florence-2
        Open Vocabulary Detection

    功能：

        1. 指定物件名稱
        2. Florence-2 偵測
        3. 取得 detection score
        4. Confidence threshold filtering
        5. Bounding Box
        6. Crop object
        7. 儲存 crop
        8. 儲存標註圖片

    回傳資料可以直接提供給 Agent 2。

    注意：
        Florence-2 的 confidence 並不是 YOLO
        那種經過校準的 objectness probability。

        這裡使用 Florence-2 generation score
        作為 filtering score。
    """

    # ======================================================
    # 初始化
    # ======================================================

    def __init__(
        self,
        model_name=MODEL_NAME,
        custom_classes=None,
        device=None
    ):

        # ==================================================
        # 1. 選擇 Device
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

        # ==================================================
        # 2. 載入 Florence-2
        # ==================================================

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
                trust_remote_code=True
            )
        )

        print(
            "✅ [Agent 1] "
            "Florence-2 載入完成"
        )

        # ==================================================
        # 3. 物件類別
        # ==================================================

        if custom_classes is None:

            self.custom_classes = [

                # ------------------------------
                # 建築 / 環境
                # ------------------------------

                "window",
                "door",
                "roof",
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

                # "tree trunk",
                # "tree base",
                # "tree branch",

                # ------------------------------
                # 其他
                # ------------------------------

                "playground tire",

            ]

        else:

            self.custom_classes = (
                custom_classes
            )

        print(
            f"✅ [Agent 1] "
            f"載入 "
            f"{len(self.custom_classes)} "
            f"種偵測標籤"
        )

    # ======================================================
    # Score → 0~1
    # ======================================================

    @staticmethod
    def _score_to_confidence(score):

        """
        將 Florence-2 score
        轉成 0~1。

        注意：
        這不是 YOLO probability。

        主要目的是方便 threshold filtering。
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
    # 單一物件 Detection
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

        # ==================================================
        # Processor
        # ==================================================

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        )

        # ==================================================
        # Move to device
        # ==================================================

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
            if hasattr(v, "to")
        }

        # ==================================================
        # Florence-2 Generate
        # ==================================================

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

                    # ⭐ 取得 token scores
                    return_dict_in_generate=True,

                    output_scores=True
                )
            )

        # ==================================================
        # Decode
        # ==================================================

        generated_text = (
            self.processor.batch_decode(
                generated.sequences,
                skip_special_tokens=False
            )[0]
        )

        # ==================================================
        # Compute transition scores
        # ==================================================

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

        # ==================================================
        # Florence-2 Post Process
        # ==================================================

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

            # ----------------------------------------------
            # 舊版 transformers 相容
            # ----------------------------------------------

            results = (
                self.processor
                .post_process_generation(
                    generated_text,
                    task=task_prompt,
                    image_size=image.size
                )
            )

            transition_scores = None

        # ==================================================
        # 取得 Detection
        # ==================================================

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

        # ==================================================
        # 取得 score
        # ==================================================

        scores = det_result.get(
            "scores",
            None
        )

        if scores is None:

            scores = det_result.get(
                "confidence",
                None
            )

        # ==================================================
        # 建立 Detection List
        # ==================================================

        detections = []

        for i, box in enumerate(
            bboxes
        ):

            # ----------------------------------------------
            # Label
            # ----------------------------------------------

            if i < len(labels):

                label = labels[i]

            else:

                label = target

            # ----------------------------------------------
            # Confidence
            # ----------------------------------------------

            confidence = None

            # 優先使用 Florence post process score
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

            # ----------------------------------------------
            # 沒有 score 時
            # 使用 transition score
            # ----------------------------------------------

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

            # ----------------------------------------------
            # Detection
            # ----------------------------------------------

            detections.append({

                "label": label,

                "box": box,

                "confidence": confidence
            })

        return detections

    # ======================================================
    # Confidence Filtering
    # ======================================================

    @staticmethod
    def _filter_by_confidence(
        detections,
        conf_threshold
    ):

        """
        Confidence filtering。

        confidence >= threshold
            → 保留

        confidence < threshold
            → 丟掉
        """

        filtered = []

        for detection in detections:

            confidence = (
                detection.get(
                    "confidence"
                )
            )

            # 沒有 confidence
            # → 保留，避免舊版相容問題

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
    # 主 Detection + Crop
    # ======================================================

    def detect_and_crop(
        self,
        image_path,
        output_dir="agent1_output",
        conf_threshold=CONF_THRESHOLD,
        imgsz=1024
    ):

        # ==================================================
        # 檢查圖片
        # ==================================================

        if not os.path.exists(
            image_path
        ):

            raise FileNotFoundError(
                f"❌ 找不到圖片："
                f"{image_path}"
            )

        # ==================================================
        # 建立 output folder
        # ==================================================

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ==================================================
        # Image name
        # ==================================================

        img_name = (
            os.path.splitext(
                os.path.basename(
                    image_path
                )
            )[0]
        )

        # ==================================================
        # OpenCV
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
        # PIL
        # ==================================================

        image = (
            Image.open(
                image_path
            ).convert("RGB")
        )

        # ==================================================
        # Start
        # ==================================================

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
            "========================================"
        )

        # ==================================================
        # All detections
        # ==================================================

        all_detections = []

        # ==================================================
        # Search each object
        # ==================================================

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

                # ------------------------------------------
                # Florence Detection
                # ------------------------------------------

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

                # ------------------------------------------
                # Confidence Filtering
                # ------------------------------------------

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

                # ------------------------------------------
                # Process each detection
                # ------------------------------------------

                for detection in filtered:

                    box = detection[
                        "box"
                    ]

                    x1, y1, x2, y2 = map(
                        int,
                        box
                    )

                    # --------------------------------------
                    # Clamp
                    # --------------------------------------

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

                    # --------------------------------------
                    # Invalid box
                    # --------------------------------------

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

                    # --------------------------------------
                    # 儲存
                    # --------------------------------------

                    all_detections.append({

                        "label": target,

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
                            confidence
                    })

                    # --------------------------------------
                    # Terminal 顯示
                    # --------------------------------------

                    if confidence is None:

                        confidence_text = (
                            "N/A"
                        )

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
        # Crop + Draw
        # ==================================================

        cropped_objects = []

        annotated_frame = (
            frame.copy()
        )

        # ==================================================
        # Process detections
        # ==================================================

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

            # ==================================================
            # Crop
            # ==================================================

            cropped_img = frame[
                y1:y2,
                x1:x2
            ]

            if cropped_img.size == 0:

                continue

            # ==================================================
            # Filename
            # ==================================================

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

            # ==================================================
            # Save crop
            # ==================================================

            cv2.imwrite(
                crop_save_path,
                cropped_img
            )

            # ==================================================
            # Draw BBox
            # ==================================================

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==================================================
            # Label + Confidence
            # ==================================================

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

            # ==================================================
            # Agent 2 Data
            # ==================================================

            cropped_objects.append({

                "id": idx,

                "label": label,

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
                    crop_save_path
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
            "\n========================================"
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

    agent1 = Agent1Detector()

    # ======================================================
    # 測試圖片
    # ======================================================

    test_image = (
        "test_images/node_21_02.jpg"
    )

    # ======================================================
    # 執行 Agent 1
    #
    # ⭐ threshold 可以直接在這裡調
    #
    # 例如：
    #
    # conf_threshold=0.05
    # conf_threshold=0.08
    # conf_threshold=0.10
    # conf_threshold=0.20
    #
    # 如果不寫：
    #
    # conf_threshold
    #
    # 就會使用最上面的：
    #
    # CONF_THRESHOLD = 0.08
    # ======================================================

    detected_items, output_image_path = (
        agent1.detect_and_crop(
            image_path=test_image,
            output_dir="agent1_output",

            # ⭐⭐⭐ 調整這裡 ⭐⭐⭐
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
            f"| confidence={confidence_text} "
            f"| BBox={item['box']} "
            f"| Crop={item['crop_path']}"
        )

