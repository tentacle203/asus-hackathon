# coarse_retriever.py

import os
import math
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class CoarseRetrieverAgent:

    """
    Coarse Retrieval Agent

    Stage 1:
        DINOv2 CLS
        80 -> Top 20

    Stage 2:
        DINOv2 Patch Matching
        20 -> Top 5

    最後提供：
        global_score
        local_score
        final_score
    """

    def __init__(
        self,
        model_name="facebook/dinov2-base",
        global_weight=0.4,
        local_weight=0.6,
        local_top_ratio=0.2
    ):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )

        print(
            f"🚀 [Coarse Agent] Device: {self.device}"
        )

        print(
            f"🚀 [Coarse Agent] Loading {model_name}"
        )

        self.processor = AutoImageProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(self.device)

        self.model.eval()

        self.global_weight = global_weight
        self.local_weight = local_weight
        self.local_top_ratio = local_top_ratio

        self.db_paths = []

        self.db_global_embeddings = None
        self.db_patch_embeddings = []

        print("✅ [Coarse Agent] DINOv2 載入完成")


    # =========================================================
    # Image loading
    # =========================================================

    def _load_image(self, img_input):

        if isinstance(img_input, str):

            if not os.path.exists(img_input):
                raise FileNotFoundError(
                    f"找不到圖片: {img_input}"
                )

            return Image.open(img_input).convert("RGB")

        elif isinstance(img_input, np.ndarray):

            return Image.fromarray(img_input).convert("RGB")

        else:

            return img_input.convert("RGB")


    # =========================================================
    # Feature extraction
    # =========================================================

    def _extract_features(self, img_input):

        pil_img = self._load_image(img_input)

        inputs = self.processor(
            images=pil_img,
            return_tensors="pt"
        )

        inputs = {
            k: v.to(self.device)
            for k, v in inputs.items()
        }

        with torch.inference_mode():

            outputs = self.model(**inputs)

            tokens = outputs.last_hidden_state

            # CLS
            global_feature = tokens[:, 0, :]

            # Patch tokens
            patch_features = tokens[:, 1:, :]

            global_feature = F.normalize(
                global_feature,
                dim=-1
            )

            patch_features = F.normalize(
                patch_features,
                dim=-1
            )

        return (
            global_feature.squeeze(0).cpu(),
            patch_features.squeeze(0).cpu()
        )


    # =========================================================
    # Build database index
    # =========================================================

    def build_db_index(self, db_folder):

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        db_paths = []

        for filename in os.listdir(db_folder):

            if filename.lower().endswith(
                valid_extensions
            ):

                db_paths.append(
                    os.path.join(
                        db_folder,
                        filename
                    )
                )

        db_paths = sorted(db_paths)

        if not db_paths:

            raise ValueError(
                f"資料庫沒有圖片: {db_folder}"
            )

        global_embeddings = []

        patch_embeddings = []

        self.db_paths = []

        print(
            f"\n📚 [Coarse Agent] 建立資料庫 index"
        )

        print(
            f"📚 圖片數量: {len(db_paths)}"
        )

        for i, path in enumerate(db_paths):

            print(
                f"   [{i + 1:03d}/{len(db_paths):03d}] "
                f"{os.path.basename(path)}"
            )

            try:

                global_emb, patch_emb = \
                    self._extract_features(path)

                global_embeddings.append(
                    global_emb
                )

                patch_embeddings.append(
                    patch_emb
                )

                self.db_paths.append(path)

            except Exception as e:

                print(
                    f"⚠️ 失敗: {path}"
                )

                print(
                    f"   {e}"
                )

        self.db_global_embeddings = torch.stack(
            global_embeddings
        )

        self.db_patch_embeddings = patch_embeddings

        print(
            "\n✅ [Coarse Agent] Index 建立完成"
        )

        print(
            f"   Global shape: "
            f"{self.db_global_embeddings.shape}"
        )

        print(
            f"   Patch images: "
            f"{len(self.db_patch_embeddings)}"
        )


    # =========================================================
    # Global retrieval
    # =========================================================

    def _global_retrieval(
        self,
        query_global,
        top_k
    ):

        query_global = query_global.unsqueeze(0)

        scores = torch.mm(
            query_global,
            self.db_global_embeddings.T
        ).squeeze(0)

        actual_k = min(
            top_k,
            len(self.db_paths)
        )

        top_scores, top_indices = torch.topk(
            scores,
            k=actual_k
        )

        candidates = []

        for score, idx in zip(
            top_scores,
            top_indices
        ):

            idx = idx.item()

            candidates.append({
                "path": self.db_paths[idx],
                "db_index": idx,
                "global_score": float(
                    score.item()
                )
            })

        return candidates


    # =========================================================
    # Local Patch Matching
    # =========================================================

    def _local_similarity(
        self,
        query_patches,
        db_patches
    ):

        # query:
        # [Nq, D]

        # database:
        # [Nd, D]

        similarity = torch.mm(
            query_patches,
            db_patches.T
        )

        # 每個 query patch
        # 找 database 中最相似的 patch

        best_matches = similarity.max(
            dim=1
        ).values

        # 取 top 20%
        num_top = max(
            1,
            int(
                len(best_matches)
                * self.local_top_ratio
            )
        )

        top_values = torch.topk(
            best_matches,
            k=num_top
        ).values

        local_score = top_values.mean()

        return float(
            local_score.item()
        )


    # =========================================================
    # Local reranking
    # =========================================================

    def _local_rerank(
        self,
        query_patches,
        candidates
    ):

        for candidate in candidates:

            db_index = candidate["db_index"]

            db_patches = \
                self.db_patch_embeddings[
                    db_index
                ]

            local_score = self._local_similarity(
                query_patches,
                db_patches
            )

            candidate["local_score"] = \
                local_score

            candidate["final_score"] = (
                self.global_weight
                * candidate["global_score"]
                +
                self.local_weight
                * candidate["local_score"]
            )

        candidates.sort(
            key=lambda x: x["final_score"],
            reverse=True
        )

        return candidates


    # =========================================================
    # Single query retrieval
    # =========================================================

    def retrieve(
        self,
        query_image,
        global_top_k=20,
        final_top_k=5
    ):

        query_global, query_patches = \
            self._extract_features(
                query_image
            )

        # -----------------------------------------------------
        # Stage 1
        # -----------------------------------------------------

        candidates = self._global_retrieval(
            query_global,
            global_top_k
        )

        # -----------------------------------------------------
        # Stage 2
        # -----------------------------------------------------

        candidates = self._local_rerank(
            query_patches,
            candidates
        )

        candidates = candidates[:final_top_k]

        return candidates


    # =========================================================
    # Four direction retrieval
    # =========================================================

    def retrieve_four_directions(
        self,
        query_images,
        global_top_k=20,
        final_top_k=5
    ):

        results = {}

        for direction, image_path in query_images.items():

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"🔎 [Coarse Agent] "
                f"{direction.upper()}"
            )

            print(
                f"📷 {image_path}"
            )

            direction_results = self.retrieve(
                image_path,
                global_top_k=global_top_k,
                final_top_k=final_top_k
            )

            results[direction] = direction_results

            for rank, result in enumerate(
                direction_results,
                start=1
            ):

                print(
                    f"{rank}. "
                    f"{os.path.basename(result['path'])}"
                )

                print(
                    f"   Global = "
                    f"{result['global_score']:.4f}"
                )

                print(
                    f"   Local  = "
                    f"{result['local_score']:.4f}"
                )

                print(
                    f"   Final  = "
                    f"{result['final_score']:.4f}"
                )

        return results