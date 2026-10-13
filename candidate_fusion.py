# candidate_fusion.py

import os
from collections import defaultdict


class CandidateFusion:

    def __init__(self):
        pass


    def fuse(
        self,
        directional_results,
        top_k=10
    ):

        candidates = defaultdict(
            lambda: {
                "path": None,
                "directions": {},
                "scores": []
            }
        )

        # =====================================================
        # 收集四方向結果
        # =====================================================

        for direction, results in \
                directional_results.items():

            for result in results:

                path = result["path"]

                data = candidates[path]

                data["path"] = path

                data["directions"][direction] = {
                    "global_score":
                        result["global_score"],

                    "local_score":
                        result["local_score"],

                    "final_score":
                        result["final_score"]
                }

                data["scores"].append(
                    result["final_score"]
                )


        # =====================================================
        # Fusion
        # =====================================================

        fused_results = []

        for path, data in candidates.items():

            scores = data["scores"]

            # 最重要：
            # 同一張照片如果被多方向找到，
            # 應該提高它的可信度。

            max_score = max(scores)

            mean_score = sum(scores) / len(scores)

            direction_count = len(scores)

            # 多方向 bonus
            direction_bonus = min(
                direction_count * 0.03,
                0.12
            )

            fused_score = (
                0.65 * max_score
                +
                0.35 * mean_score
                +
                direction_bonus
            )

            fused_results.append({

                "path": path,

                "fused_score": fused_score,

                "max_score": max_score,

                "mean_score": mean_score,

                "direction_count":
                    direction_count,

                "directions":
                    data["directions"]
            })


        fused_results.sort(
            key=lambda x: x["fused_score"],
            reverse=True
        )

        return fused_results[:top_k]