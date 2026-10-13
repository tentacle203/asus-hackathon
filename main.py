
# main.py

import os

from PIL import Image

from config import *

from database_metadata import DatabaseMetadata
from coarse_retriever import CoarseRetrieverAgent
from candidate_fusion import CandidateFusion

from agent1_detector import Agent1Detector
from agent2_matcher import Agent2Matcher
from agent3_localizer import Agent3Localizer


# =========================================================
# Helper
# =========================================================

def print_section(title):

    print("\n")
    print("=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# Main
# =========================================================

def main():

    print_section(
        "🚀 Multi-Agent Outdoor Localization System"
    )

    # =====================================================
    # 1. Query Images
    # =====================================================

    query_images = {

        "east":
            os.path.join(
                QUERY_FOLDER,
                "node_21_02.jpg"
            ),

        "west":
            os.path.join(
                QUERY_FOLDER,
                "node_21_04.jpg"
            ),

        "south":
            os.path.join(
                QUERY_FOLDER,
                "node_21_03.jpg"
            ),

        "north":
            os.path.join(
                QUERY_FOLDER,
                "node_21_01.jpg"
            )
    }

    print_section(
        "📷 Step 1 — Check Query Images"
    )

    for direction, path in query_images.items():

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"❌ 找不到 "
                f"{direction} query image:\n"
                f"{path}"
            )

        print(
            f"✅ {direction:6s}: "
            f"{path}"
        )

    # =====================================================
    # 2. Load Database Metadata
    # =====================================================

    print_section(
        "📍 Step 2 — Load Database GPS Metadata"
    )

    metadata = DatabaseMetadata(
        METADATA_PATH
    )

    print(
        f"✅ Metadata loaded: "
        f"{len(metadata.metadata)} images"
    )

    # =====================================================
    # 3. Initialize Coarse Retriever
    # =====================================================

    print_section(
        "🔎 Step 3 — Initialize Coarse Retrieval Agent"
    )

    coarse_agent = CoarseRetrieverAgent(

        model_name=DINO_MODEL,

        global_weight=GLOBAL_WEIGHT,

        local_weight=LOCAL_WEIGHT,

        local_top_ratio=LOCAL_TOP_RATIO
    )

    # =====================================================
    # 4. Build Database Index
    # =====================================================

    print_section(
        "📚 Step 4 — Build DINOv2 Database Index"
    )

    coarse_agent.build_db_index(
        DATABASE_FOLDER
    )

    # =====================================================
    # 5. Four Direction Coarse Retrieval
    # =====================================================

    print_section(
        "🧭 Step 5 — Four Direction Coarse Retrieval"
    )

    directional_results = \
        coarse_agent.retrieve_four_directions(

            query_images,

            global_top_k=GLOBAL_TOP_K,

            final_top_k=LOCAL_TOP_K
        )

    # =====================================================
    # 6. Candidate Fusion
    # =====================================================

    print_section(
        "🔗 Step 6 — Candidate Fusion"
    )

    fusion = CandidateFusion()

    fused_candidates = fusion.fuse(

        directional_results,

        top_k=10
    )

    print(
        "\n🏆 Fused Candidates:"
    )

    for rank, candidate in enumerate(
        fused_candidates,
        start=1
    ):

        print(
            f"\n#{rank} "
            f"{os.path.basename(candidate['path'])}"
        )

        print(
            f"   fused_score = "
            f"{candidate['fused_score']:.4f}"
        )

        print(
            f"   directions = "
            f"{list(candidate['directions'].keys())}"
        )

    # =====================================================
    # 7. Initialize Agent 1
    # =====================================================

    print_section(
        "🤖 Step 7 — Initialize Agent 1"
    )

    agent1 = Agent1Detector()

    # =====================================================
    # 8. Run Agent 1
    #
    # 重要：
    # 每個 Query 只做一次 detection
    # 不要在每個 DB candidate 裡重複做
    # =====================================================

    print_section(
        "🔬 Step 8 — Agent 1 Object Detection"
    )

    all_detections = {}

    for direction, query_path in query_images.items():

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"🧭 Agent 1 — "
            f"{direction.upper()}"
        )

        detections, annotated_img_path = \
            agent1.detect_and_crop(

                query_path,

                conf_threshold=CONF_THRESHOLD,

                output_dir=OUTPUT_FOLDER
            )

        all_detections[direction] = \
            detections

        print(
            f"   detections = "
            f"{len(detections)}"
        )

        print(
            f"   annotated = "
            f"{annotated_img_path}"
        )

    # =====================================================
    # 9. Initialize Agent 2
    # =====================================================

    print_section(
        "🧠 Step 9 — Initialize Agent 2"
    )

    agent2 = Agent2Matcher()

    # =====================================================
    # 10. Agent 2 Fine Matching
    # =====================================================

    print_section(
        "🔬 Step 10 — Agent 2 Fine Matching"
    )

    agent2_results = []

    # -----------------------------------------------------
    # 每一個 coarse candidate
    # ×
    # 四個 query directions
    # -----------------------------------------------------

    for candidate_index, candidate in enumerate(
        fused_candidates,
        start=1
    ):

        db_path = candidate["path"]

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"📷 Candidate "
            f"{candidate_index}/"
            f"{len(fused_candidates)}"
        )

        print(
            f"   {os.path.basename(db_path)}"
        )

        print(
            f"   Coarse fused score = "
            f"{candidate['fused_score']:.4f}"
        )

        # =================================================
        # 四個方向
        # =================================================

        for direction, query_path in \
                query_images.items():

            print(
                f"\n   🧭 "
                f"{direction.upper()}"
            )

            detections = \
                all_detections[
                    direction
                ]

            # -------------------------------------------------
            # Agent 2
            # -------------------------------------------------

            result = agent2.match(

                query_image=query_path,

                database_image=db_path,

                detections=detections,

                direction=direction
            )

            # -------------------------------------------------
            # 加入 Agent 3 需要的資訊
            # -------------------------------------------------

            if not isinstance(
                result,
                dict
            ):

                print(
                    "   ⚠️ Agent 2 "
                    "result 不是 dict"
                )

                continue

            result["image"] = db_path

            result["direction"] = \
                direction

            result["coarse_fused_score"] = \
                candidate[
                    "fused_score"
                ]

            result["coarse_candidate_rank"] = \
                candidate_index

            # -------------------------------------------------
            # 儲存
            # -------------------------------------------------

            agent2_results.append(
                result
            )

            # -------------------------------------------------
            # Terminal
            # -------------------------------------------------

            print(
                f"      similarity = "
                f"{result.get('similarity_score', 0):.4f}"
            )

            print(
                f"      ROI count = "
                f"{result.get('num_roi', 0)}"
            )

    # =====================================================
    # 11. Agent 2 Summary
    # =====================================================

    print_section(
        "📊 Step 11 — Agent 2 Result Summary"
    )

    print(
        f"Total Agent 2 matches: "
        f"{len(agent2_results)}"
    )

    # -----------------------------------------------------
    # 按 similarity 排序
    # -----------------------------------------------------

    sorted_agent2_results = sorted(

        agent2_results,

        key=lambda x:
            x.get(
                "similarity_score",
                0.0
            ),

        reverse=True
    )

    print(
        "\nTop Agent 2 Matches:"
    )

    for rank, result in enumerate(
        sorted_agent2_results[:10],
        start=1
    ):

        print(
            f"\n#{rank}"
        )

        print(
            f"   image = "
            f"{os.path.basename(result['image'])}"
        )

        print(
            f"   direction = "
            f"{result.get('direction')}"
        )

        print(
            f"   similarity = "
            f"{result.get('similarity_score', 0):.4f}"
        )

        print(
            f"   ROI = "
            f"{result.get('num_roi', 0)}"
        )

    # =====================================================
    # 12. Prepare Image Sizes
    #
    # Agent 3 需要知道 bbox 在原圖中的位置
    # =====================================================

    print_section(
        "📐 Step 12 — Read Query Image Sizes"
    )

    image_sizes = {}

    for direction, query_path in \
            query_images.items():

        with Image.open(
            query_path
        ) as img:

            image_sizes[direction] = \
                img.size

        print(
            f"   {direction:6s}: "
            f"{image_sizes[direction]}"
        )

    # =====================================================
    # 13. Initialize Agent 3
    # =====================================================

    print_section(
        "📍 Step 13 — Initialize Agent 3"
    )

    agent3 = Agent3Localizer(

        metadata=metadata,

        visual_weight=0.85,

        heading_weight=0.15,

        # -------------------------------------------------
        # 相機水平 FOV
        #
        # 如果之後知道實際相機 FOV，
        # 可以直接修改這裡
        # -------------------------------------------------

        horizontal_fov=90.0,

        # -------------------------------------------------
        # 地圖上 ray 顯示長度
        # 單位：meter
        # -------------------------------------------------

        ray_length=200.0
    )

    # =====================================================
    # 14. Agent 3 Geometry Localization
    # =====================================================

    print_section(
        "📍 Step 14 — Agent 3 Geometry Localization"
    )

    gps_result = agent3.localize(

        matched_candidates=
            agent2_results,

        image_sizes=
            image_sizes,

        output_map=
            os.path.join(
                OUTPUT_FOLDER,
                "agent3_map.html"
            )
    )

    # =====================================================
    # 15. Final Result
    # =====================================================

    print_section(
        "🎯 FINAL RESULT"
    )

    if gps_result is None:

        print(
            "❌ Agent 3 無法產生定位結果"
        )

        return

    print(
        f"Latitude  : "
        f"{gps_result['latitude']:.8f}"
    )

    print(
        f"Longitude : "
        f"{gps_result['longitude']:.8f}"
    )

    print(
        f"Confidence: "
        f"{gps_result['confidence']:.4f}"
    )

    print(
        f"Mean Error: "
        f"{gps_result['mean_intersection_error_m']:.2f} m"
    )

    print(
        f"Max Error : "
        f"{gps_result['max_intersection_error_m']:.2f} m"
    )

    print(
        f"Number of Rays: "
        f"{gps_result['num_rays']}"
    )

    if gps_result.get("map"):

        print(
            "\n🗺️ Interactive Map:"
        )

        print(
            f"   "
            f"{gps_result['map']['map_path']}"
        )

        print(
            "\n💡 Open with:"
        )

        print(
            f"   open "
            f"{gps_result['map']['map_path']}"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "✅ Multi-Agent Localization Finished"
    )

    print(
        "=" * 80
    )


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()

