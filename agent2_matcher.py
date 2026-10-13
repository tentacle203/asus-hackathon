
# agent2_matcher.py

import os
import math
from typing import List, Dict, Optional

import torch
import torch.nn.functional as F

from PIL import Image
from transformers import AutoProcessor, CLIPModel


class Agent2Matcher:
    """
    ============================================================
    Agent 2
    ============================================================

    CLIP-based Fine Image Matching Agent

    接收：

        Coarse Agent
            ↓
        Candidate database images

        Agent 1
            ↓
        Query image ROI detections

    然後進行：

        1. Query 整張圖 ↔ Database 整張圖
        2. Agent 1 ROI ↔ Database 整張圖
        3. 聚合所有 ROI similarity
        4. 產生 final_score

    最後輸出給 Agent 3：

        {
            "image": database_image,
            "similarity_score": ...,
            "direction": ...,
            "details": ...
        }


    注意：

    目前你的 database 只有：

        image
        latitude
        longitude
        heading

    沒有 database ROI。

    所以目前採用：

        Query ROI
            ↕
        Database Full Image

    如果之後你也讓 80 張 database 圖片跑 Agent 1，
    就可以再升級成：

        Query ROI ↔ Database ROI
    """


    def __init__(
        self,
        model_id="openai/clip-vit-base-patch32",
        device=None,

        # Agent 2 最低接受分數
        match_threshold=0.70,

        # Full image weight
        full_image_weight=0.30,

        # ROI weight
        roi_weight=0.70,

        # 多個 ROI 中取最高幾個
        roi_top_k=5,

        # 是否使用 ROI label
        use_label_bonus=True,

        # label bonus
        label_bonus_weight=0.05
    ):

        # =====================================================
        # Device
        # =====================================================

        if device is None:

            if torch.backends.mps.is_available():

                self.device = "mps"

            elif torch.cuda.is_available():

                self.device = "cuda"

            else:

                self.device = "cpu"

        else:

            self.device = device


        print(
            f"🔍 [Agent 2] 初始化 CLIP "
            f"使用裝置: {self.device}"
        )


        # =====================================================
        # Model
        # =====================================================

        self.model = CLIPModel.from_pretrained(
            model_id
        ).to(
            self.device
        ).eval()


        self.processor = AutoProcessor.from_pretrained(
            model_id
        )


        # =====================================================
        # Parameters
        # =====================================================

        self.match_threshold = \
            match_threshold

        self.full_image_weight = \
            full_image_weight

        self.roi_weight = \
            roi_weight

        self.roi_top_k = \
            roi_top_k

        self.use_label_bonus = \
            use_label_bonus

        self.label_bonus_weight = \
            label_bonus_weight


        # =====================================================
        # Database feature cache
        # =====================================================

        self.database_feature_cache = {}


        print(
            "✅ [Agent 2] CLIP Matching Agent 載入完成"
        )


    # =========================================================
    # Image loading
    # =========================================================

    def load_image(
        self,
        image_input
    ) -> Image.Image:

        if isinstance(
            image_input,
            Image.Image
        ):

            return image_input.convert(
                "RGB"
            )


        if isinstance(
            image_input,
            str
        ):

            if not os.path.exists(
                image_input
            ):

                raise FileNotFoundError(
                    f"❌ 找不到圖片: "
                    f"{image_input}"
                )


            return Image.open(
                image_input
            ).convert(
                "RGB"
            )


        raise TypeError(
            "image_input 必須是 "
            "PIL.Image 或圖片路徑"
        )


    # =========================================================
    # Extract CLIP image feature
    # =========================================================

    def extract_features(
        self,
        image: Image.Image
    ) -> torch.Tensor:

        image = image.convert(
            "RGB"
        )


        inputs = self.processor(
            images=image,
            return_tensors="pt"
        )


        inputs = {
            key: value.to(
                self.device
            )

            for key, value in inputs.items()
        }


        with torch.inference_mode():

            image_features = \
                self.model.get_image_features(
                    **inputs
                )


        image_features = F.normalize(
            image_features,
            dim=-1
        )


        return image_features


    # =========================================================
    # Compare feature vectors
    # =========================================================

    def compare_features(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> float:

        similarity = torch.mm(
            feat1,
            feat2.T
        ).item()


        return float(
            similarity
        )


    # =========================================================
    # Compare two images
    # =========================================================

    def compare_images(
        self,
        img1: Image.Image,
        img2: Image.Image
    ) -> float:

        feat1 = self.extract_features(
            img1
        )

        feat2 = self.extract_features(
            img2
        )


        return self.compare_features(
            feat1,
            feat2
        )


    # =========================================================
    # Get database feature
    #
    # 重要：
    # Candidate database image 不應該每次重新算 CLIP
    #
    # =========================================================

    def get_database_feature(
        self,
        image_path: str
    ):

        if image_path in \
                self.database_feature_cache:

            return self.database_feature_cache[
                image_path
            ]


        image = self.load_image(
            image_path
        )


        feature = self.extract_features(
            image
        )


        self.database_feature_cache[
            image_path
        ] = feature


        return feature


    # =========================================================
    # Preload database candidates
    # =========================================================

    def preload_database(
        self,
        image_paths: List[str]
    ):

        print(
            "\n📚 [Agent 2] "
            "預先計算 Candidate CLIP features"
        )


        for i, image_path in enumerate(
            image_paths,
            start=1
        ):

            print(
                f"   [{i}/{len(image_paths)}] "
                f"{os.path.basename(image_path)}"
            )


            self.get_database_feature(
                image_path
            )


        print(
            "✅ [Agent 2] "
            "Candidate features 完成"
        )


    # =========================================================
    # Crop Agent 1 ROI
    # =========================================================

    def crop_detection(
        self,
        main_image: Image.Image,
        detection: Dict
    ) -> Optional[Image.Image]:

        bbox = detection.get(
            "box"
        )


        if bbox is None:

            return None


        if len(bbox) != 4:

            return None


        width, height = \
            main_image.size


        x1, y1, x2, y2 = bbox


        # -----------------------------------------------------
        # Clamp
        # -----------------------------------------------------

        x1 = max(
            0,
            min(
                int(x1),
                width - 1
            )
        )

        y1 = max(
            0,
            min(
                int(y1),
                height - 1
            )
        )

        x2 = max(
            x1 + 1,
            min(
                int(x2),
                width
            )
        )

        y2 = max(
            y1 + 1,
            min(
                int(y2),
                height
            )
        )


        # -----------------------------------------------------
        # Crop
        # -----------------------------------------------------

        crop = main_image.crop(
            (
                x1,
                y1,
                x2,
                y2
            )
        )


        return crop


    # =========================================================
    # ROI similarity
    # =========================================================

    def calculate_roi_scores(
        self,
        query_image: Image.Image,
        detections: List[Dict],
        database_feature: torch.Tensor
    ):

        roi_results = []


        for detection in detections:

            crop = self.crop_detection(
                query_image,
                detection
            )


            if crop is None:

                continue


            # -------------------------------------------------
            # 過濾太小 ROI
            # -------------------------------------------------

            width, height = crop.size


            if width < 10 or height < 10:

                continue


            # -------------------------------------------------
            # CLIP feature
            # -------------------------------------------------

            crop_feature = \
                self.extract_features(
                    crop
                )


            similarity = \
                self.compare_features(
                    crop_feature,
                    database_feature
                )


            label = (
                detection.get(
                    "detected_label"
                )

                or

                detection.get(
                    "label"
                )

                or

                "unknown"
            )


            confidence = \
                detection.get(
                    "confidence"
                )


            roi_results.append({

                "label":
                    label,

                "similarity":
                    similarity,

                "detection_confidence":
                    confidence,

                "box":
                    detection.get(
                        "box"
                    ),

                "crop":
                    crop
            })


        return roi_results


    # =========================================================
    # Aggregate ROI scores
    # =========================================================

    def aggregate_roi_scores(
        self,
        roi_results: List[Dict]
    ) -> float:

        if not roi_results:

            return 0.0


        # -----------------------------------------------------
        # 由高到低排序
        # -----------------------------------------------------

        sorted_results = sorted(

            roi_results,

            key=lambda x:
                x["similarity"],

            reverse=True
        )


        # -----------------------------------------------------
        # 取 Top-K ROI
        # -----------------------------------------------------

        top_results = sorted_results[
            :self.roi_top_k
        ]


        # -----------------------------------------------------
        # 計算加權平均
        #
        # 高 similarity 的 ROI
        # 權重稍微高一點
        # -----------------------------------------------------

        weights = []


        for result in top_results:

            score = result[
                "similarity"
            ]


            # similarity [-1, 1]
            #
            # 轉到 [0, 1]

            normalized = (
                score + 1.0
            ) / 2.0


            weight = max(
                normalized,
                0.001
            )


            weights.append(
                weight
            )


        total_weight = sum(
            weights
        )


        if total_weight <= 0:

            return 0.0


        weighted_score = sum(

            result["similarity"]
            * weight

            for result, weight
            in zip(
                top_results,
                weights
            )

        ) / total_weight


        return float(
            weighted_score
        )


    # =========================================================
    # Full image similarity
    # =========================================================

    def calculate_full_image_score(
        self,
        query_image: Image.Image,
        database_feature: torch.Tensor
    ) -> float:

        query_feature = \
            self.extract_features(
                query_image
            )


        return self.compare_features(
            query_feature,
            database_feature
        )


    # =========================================================
    # Normalize similarity
    # =========================================================

    def normalize_score(
        self,
        score: float
    ) -> float:

        """
        CLIP cosine:

            -1 ~ +1

        轉成：

             0 ~ 1
        """

        score = float(
            score
        )


        normalized = (
            score + 1.0
        ) / 2.0


        return max(
            0.0,
            min(
                1.0,
                normalized
            )
        )


    # =========================================================
    # Calculate final score
    # =========================================================

    def calculate_final_score(
        self,
        full_score: float,
        roi_score: float,
        num_roi: int
    ) -> float:

        full_normalized = \
            self.normalize_score(
                full_score
            )


        roi_normalized = \
            self.normalize_score(
                roi_score
            )


        # -----------------------------------------------------
        # 如果沒有 ROI
        #
        # 就只使用 full image
        # -----------------------------------------------------

        if num_roi == 0:

            return full_normalized


        # -----------------------------------------------------
        # Full + ROI
        # -----------------------------------------------------

        final_score = (

            self.full_image_weight
            * full_normalized

            +

            self.roi_weight
            * roi_normalized
        )


        return float(
            final_score
        )


    # =========================================================
    # Main Matching Function
    # =========================================================

    def match(
        self,
        query_image,
        database_image: str,
        detections: Optional[List[Dict]] = None,
        direction: Optional[str] = None
    ) -> Dict:

        """
        ========================================================
        Agent 2 Main API
        ========================================================

        這個函數就是給 main.py 呼叫的。

        Example:

            result = agent2.match(

                query_image="query/north.jpg",

                database_image="data/database/023.jpg",

                detections=agent1_results,

                direction="north"
            )


        Parameters
        ----------

        query_image:
            Query 圖片

        database_image:
            Coarse Agent 找出的 candidate

        detections:
            Agent 1 detect_and_crop() 回傳結果

        direction:
            east / west / south / north


        Returns
        -------

        {
            "image": "...",

            "similarity_score": 0.91,

            "is_match": True,

            "direction": "north",

            "full_image_score": ...,

            "roi_score": ...,

            "roi_results": [...]
        }
        """


        # =====================================================
        # Load images
        # =====================================================

        query_img = self.load_image(
            query_image
        )


        database_img = self.load_image(
            database_image
        )


        # =====================================================
        # Database feature
        # =====================================================

        database_feature = \
            self.get_database_feature(
                database_image
            )


        # =====================================================
        # Full Image Matching
        # =====================================================

        full_score = \
            self.calculate_full_image_score(

                query_img,

                database_feature
            )


        # =====================================================
        # ROI Matching
        # =====================================================

        if detections is None:

            detections = []


        roi_results = \
            self.calculate_roi_scores(

                query_img,

                detections,

                database_feature
            )


        # =====================================================
        # ROI aggregation
        # =====================================================

        roi_score = \
            self.aggregate_roi_scores(
                roi_results
            )


        # =====================================================
        # Final score
        # =====================================================

        final_score = \
            self.calculate_final_score(

                full_score,

                roi_score,

                len(roi_results)
            )


        # =====================================================
        # Match threshold
        # =====================================================

        is_match = (
            final_score
            >=
            self.match_threshold
        )


        # =====================================================
        # Output
        # =====================================================

        result = {

            # Agent 3 最重要
            "image":
                database_image,

            "similarity_score":
                round(
                    final_score,
                    4
                ),

            "is_match":
                is_match,

            "direction":
                direction,

            # Debug
            "full_image_score":
                round(
                    full_score,
                    4
                ),

            "roi_score":
                round(
                    roi_score,
                    4
                ),

            "num_roi":
                len(
                    roi_results
                ),

            "roi_results":
                [
                    {
                        "label":
                            item["label"],

                        "similarity":
                            round(
                                item[
                                    "similarity"
                                ],
                                4
                            ),

                        "detection_confidence":
                            item[
                                "detection_confidence"
                            ],

                        "box":
                            item["box"]
                    }

                    for item
                    in roi_results
                ]
        }


        # =====================================================
        # Print result
        # =====================================================

        print(
            "\n"
            f"🔎 [Agent 2] "
            f"{os.path.basename(database_image)}"
        )


        print(
            f"   Direction     : "
            f"{direction}"
        )


        print(
            f"   Full image   : "
            f"{full_score:.4f}"
        )


        print(
            f"   ROI score    : "
            f"{roi_score:.4f}"
        )


        print(
            f"   ROI count    : "
            f"{len(roi_results)}"
        )


        print(
            f"   Final score  : "
            f"{final_score:.4f}"
        )


        print(
            f"   Match        : "
            f"{'✅' if is_match else '❌'}"
        )


        # -----------------------------------------------------
        # 印出 ROI
        # -----------------------------------------------------

        if roi_results:

            print(
                "   ROI details:"
            )


            sorted_roi = sorted(

                roi_results,

                key=lambda x:
                    x["similarity"],

                reverse=True
            )


            for roi in sorted_roi:

                print(

                    f"      "
                    f"{roi['label']}: "
                    f"{roi['similarity']:.4f}"
                )


        return result


    # =========================================================
    # Batch matching
    # =========================================================

    def match_candidates(
        self,
        query_image,
        candidates: List[str],
        detections=None,
        direction=None
    ) -> List[Dict]:

        """
        對 Coarse Agent 找出的多張 candidate
        一次進行 Agent 2。

        Example:

            candidates = [
                "001.jpg",
                "023.jpg",
                "024.jpg",
                "031.jpg"
            ]

        """

        results = []


        for database_image in candidates:

            result = self.match(

                query_image=query_image,

                database_image=database_image,

                detections=detections,

                direction=direction
            )


            results.append(
                result
            )


        # -----------------------------------------------------
        # 依 similarity 排序
        # -----------------------------------------------------

        results.sort(

            key=lambda x:
                x["similarity_score"],

            reverse=True
        )


        return results


    # =========================================================
    # Get best candidate
    # =========================================================

    def get_best_candidate(
        self,
        results: List[Dict]
    ) -> Optional[Dict]:

        if not results:

            return None


        return max(

            results,

            key=lambda x:
                x["similarity_score"]
        )


# =============================================================
# Standalone Test
# =============================================================

if __name__ == "__main__":

    print(
        "=" * 70
    )

    print(
        "🧪 Agent 2 Standalone Test"
    )

    print(
        "=" * 70
    )


    agent2 = Agent2Matcher()


    # ---------------------------------------------------------
    # 測試圖片
    # ---------------------------------------------------------

    query_path = "query/north.jpg"

    database_path = \
        "data/database/023.jpg"


    if (
        os.path.exists(query_path)
        and
        os.path.exists(database_path)
    ):

        result = agent2.match(

            query_image=query_path,

            database_image=database_path,

            detections=[],

            direction="north"
        )


        print(
            "\n📊 Final Result:"
        )


        print(
            f"Image: "
            f"{result['image']}"
        )


        print(
            f"Score: "
            f"{result['similarity_score']}"
        )


        print(
            f"Match: "
            f"{result['is_match']}"
        )

    else:

        print(
            "\n⚠️ 找不到測試圖片"
        )

        print(
            f"Query: {query_path}"
        )

        print(
            f"Database: {database_path}"
        )

