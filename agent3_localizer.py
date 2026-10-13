
# agent3_localizer.py

import os
import math
import numpy as np
import folium


class Agent3Localizer:

    def __init__(
        self,
        metadata,
        visual_weight=0.85,
        heading_weight=0.15,
        horizontal_fov=90.0,
        ray_length=200.0
    ):
        """
        Agent 3:
        使用 Agent 2 的視覺匹配結果，
        搭配 Database GPS、heading、ROI bbox
        進行幾何定位。

        Parameters
        ----------
        metadata:
            DatabaseMetadata instance

        visual_weight:
            Agent 2 visual similarity 權重

        heading_weight:
            heading similarity 權重

        horizontal_fov:
            相機水平 FOV，單位 degree

        ray_length:
            地圖上畫出的視線長度，單位 meter
        """

        self.metadata = metadata

        self.visual_weight = visual_weight
        self.heading_weight = heading_weight

        self.horizontal_fov = horizontal_fov
        self.ray_length = ray_length

    # =========================================================
    # 1. Heading → degree
    # =========================================================

    def heading_to_degree(self, heading):

        if heading is None:
            return None

        # 已經是數字
        if isinstance(heading, (int, float)):
            return float(heading) % 360.0

        heading = str(heading).lower().strip()

        mapping = {
            "north": 0.0,
            "n": 0.0,

            "east": 90.0,
            "e": 90.0,

            "south": 180.0,
            "s": 180.0,

            "west": 270.0,
            "w": 270.0
        }

        if heading in mapping:
            return mapping[heading]

        try:
            return float(heading) % 360.0

        except Exception:
            return None

    # =========================================================
    # 2. Heading similarity
    # =========================================================

    def heading_similarity(
        self,
        query_heading,
        db_heading
    ):

        q = self.heading_to_degree(
            query_heading
        )

        d = self.heading_to_degree(
            db_heading
        )

        if q is None or d is None:
            return 0.0

        diff = abs(q - d)

        diff = min(
            diff,
            360.0 - diff
        )

        # 0°差異 → 1
        # 180°差異 → 0
        score = (
            math.cos(
                math.radians(diff)
            ) + 1.0
        ) / 2.0

        return score

    # =========================================================
    # 3. GPS → Local XY
    #
    # x = East
    # y = North
    #
    # 單位：meter
    # =========================================================

    def gps_to_xy(
        self,
        latitude,
        longitude,
        origin_lat,
        origin_lon
    ):

        R = 6371000.0

        lat_rad = math.radians(
            origin_lat
        )

        x = (
            math.radians(
                longitude - origin_lon
            )
            * R
            * math.cos(lat_rad)
        )

        y = (
            math.radians(
                latitude - origin_lat
            )
            * R
        )

        return np.array([
            x,
            y
        ], dtype=float)

    # =========================================================
    # 4. Local XY → GPS
    # =========================================================

    def xy_to_gps(
        self,
        x,
        y,
        origin_lat,
        origin_lon
    ):

        R = 6371000.0

        lat_rad = math.radians(
            origin_lat
        )

        latitude = (
            origin_lat
            + math.degrees(
                y / R
            )
        )

        longitude = (
            origin_lon
            + math.degrees(
                x
                / (
                    R
                    * math.cos(lat_rad)
                )
            )
        )

        return (
            latitude,
            longitude
        )

    # =========================================================
    # 5. Bbox → image horizontal angle
    #
    # image center = 0°
    #
    # 左邊 = negative
    # 右邊 = positive
    # =========================================================

    def bbox_to_horizontal_angle(
        self,
        bbox,
        image_width
    ):

        if bbox is None:
            return 0.0

        if image_width is None:
            return 0.0

        x1, y1, x2, y2 = bbox

        center_x = (
            x1 + x2
        ) / 2.0

        image_center = (
            image_width
            / 2.0
        )

        if image_center == 0:
            return 0.0

        normalized = (
            center_x - image_center
        ) / image_center

        # 限制在 [-1, 1]
        normalized = max(
            -1.0,
            min(
                1.0,
                normalized
            )
        )

        relative_angle = (
            normalized
            * self.horizontal_fov
            / 2.0
        )

        return relative_angle

    # =========================================================
    # 6. DB heading + ROI angle
    #
    # 得到真正的世界 bearing
    # =========================================================

    def calculate_world_bearing(
        self,
        db_heading,
        relative_angle
    ):

        heading = self.heading_to_degree(
            db_heading
        )

        if heading is None:
            return None

        bearing = (
            heading
            + relative_angle
        ) % 360.0

        return bearing

    # =========================================================
    # 7. Bearing → XY direction
    #
    # bearing:
    #
    # 0°   = North
    # 90°  = East
    # 180° = South
    # 270° = West
    # =========================================================

    def bearing_to_direction(
        self,
        bearing
    ):

        rad = math.radians(
            bearing
        )

        dx = math.sin(rad)

        dy = math.cos(rad)

        vector = np.array(
            [
                dx,
                dy
            ],
            dtype=float
        )

        norm = np.linalg.norm(
            vector
        )

        if norm == 0:
            return None

        return vector / norm

    # =========================================================
    # 8. 從 Agent 2 result 找最佳 ROI
    # =========================================================

    def get_best_roi(
        self,
        result
    ):

        roi_results = result.get(
            "roi_results",
            []
        )

        if not roi_results:
            return None

        best_roi = max(
            roi_results,
            key=lambda x:
                x.get(
                    "similarity",
                    -1.0
                )
        )

        return best_roi

    # =========================================================
    # 9. 建立 viewing rays
    # =========================================================

    def build_rays(
        self,
        matched_results,
        image_sizes
    ):

        rays = []

        for result in matched_results:

            image_path = result.get(
                "image"
            )

            if image_path is None:
                continue

            # -------------------------------------------------
            # Database metadata
            # -------------------------------------------------

            metadata = self.metadata.get(
                image_path
            )

            if metadata is None:

                print(
                    f"⚠️ 找不到 metadata:"
                    f" {image_path}"
                )

                continue

            latitude = metadata.get(
                "latitude"
            )

            longitude = metadata.get(
                "longitude"
            )

            heading = metadata.get(
                "heading"
            )

            if latitude is None:
                continue

            if longitude is None:
                continue

            if heading is None:
                continue

            # -------------------------------------------------
            # Query direction
            # -------------------------------------------------

            direction_name = result.get(
                "direction"
            )

            # -------------------------------------------------
            # Query image size
            # -------------------------------------------------

            if direction_name not in image_sizes:

                print(
                    f"⚠️ 找不到 image size:"
                    f" {direction_name}"
                )

                continue

            image_width, image_height = \
                image_sizes[
                    direction_name
                ]

            # -------------------------------------------------
            # Agent 2 similarity
            # -------------------------------------------------

            similarity_score = result.get(
                "similarity_score",
                0.0
            )

            # -------------------------------------------------
            # 找 Agent 2 最佳 ROI
            # -------------------------------------------------

            best_roi = self.get_best_roi(
                result
            )

            if best_roi is None:

                print(
                    "⚠️ Agent 2 沒有 ROI:"
                    f" {image_path}"
                )

                continue

            bbox = best_roi.get(
                "box"
            )

            roi_label = best_roi.get(
                "label",
                "unknown"
            )

            roi_similarity = best_roi.get(
                "similarity",
                0.0
            )

            if bbox is None:
                continue

            # -------------------------------------------------
            # bbox → 相對水平角
            # -------------------------------------------------

            relative_angle = \
                self.bbox_to_horizontal_angle(

                    bbox=bbox,

                    image_width=image_width
                )

            # -------------------------------------------------
            # DB heading + ROI angle
            # -------------------------------------------------

            bearing = \
                self.calculate_world_bearing(

                    db_heading=heading,

                    relative_angle=relative_angle
                )

            if bearing is None:
                continue

            # -------------------------------------------------
            # bearing → direction vector
            # -------------------------------------------------

            direction_vector = \
                self.bearing_to_direction(
                    bearing
                )

            if direction_vector is None:
                continue

            # -------------------------------------------------
            # 儲存 ray
            # -------------------------------------------------

            rays.append({

                "image":
                    image_path,

                "latitude":
                    float(latitude),

                "longitude":
                    float(longitude),

                "heading":
                    self.heading_to_degree(
                        heading
                    ),

                "direction_name":
                    direction_name,

                "relative_angle":
                    relative_angle,

                "bearing":
                    bearing,

                "direction_vector":
                    direction_vector,

                "similarity_score":
                    similarity_score,

                "roi_label":
                    roi_label,

                "roi_similarity":
                    roi_similarity,

                "bbox":
                    bbox

            })

        return rays

    # =========================================================
    # 10. Ray intersection
    #
    # 找到最符合所有 viewing rays 的點
    # =========================================================

    def calculate_ray_intersection(
        self,
        rays,
        origin_lat,
        origin_lon
    ):

        if len(rays) < 2:

            print(
                "⚠️ 至少需要 2 條 ray"
                " 才能計算 intersection"
            )

            return None

        origins = []
        directions = []

        for ray in rays:

            p = self.gps_to_xy(

                latitude=ray["latitude"],

                longitude=ray["longitude"],

                origin_lat=origin_lat,

                origin_lon=origin_lon
            )

            d = np.asarray(
                ray["direction_vector"],
                dtype=float
            )

            norm = np.linalg.norm(d)

            if norm == 0:
                continue

            d = d / norm

            origins.append(p)
            directions.append(d)

        if len(origins) < 2:
            return None

        # -----------------------------------------------------
        # Least-squares line intersection
        #
        # 對每條 ray：
        #
        # M = I - dd^T
        #
        # A += M
        # b += M p
        #
        # A x = b
        # -----------------------------------------------------

        A = np.zeros(
            (2, 2),
            dtype=float
        )

        b = np.zeros(
            2,
            dtype=float
        )

        I = np.eye(2)

        for p, d in zip(
            origins,
            directions
        ):

            M = (
                I
                - np.outer(
                    d,
                    d
                )
            )

            A += M

            b += M @ p

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------

        try:

            point = np.linalg.solve(
                A,
                b
            )

        except np.linalg.LinAlgError:

            print(
                "⚠️ Ray intersection "
                "矩陣無法求解"
            )

            return None

        # -----------------------------------------------------
        # 計算每條 ray 到 intersection
        # 的垂直距離
        # -----------------------------------------------------

        distances = []

        forward_distances = []

        for p, d in zip(
            origins,
            directions
        ):

            diff = point - p

            # projection distance
            t = np.dot(
                diff,
                d
            )

            # 最近點
            closest = (
                p + t * d
            )

            distance = np.linalg.norm(
                point - closest
            )

            distances.append(
                distance
            )

            forward_distances.append(
                t
            )

        mean_error = float(
            np.mean(
                distances
            )
        )

        max_error = float(
            np.max(
                distances
            )
        )

        return {

            "point":
                point,

            "mean_error":
                mean_error,

            "max_error":
                max_error,

            "distances":
                distances,

            "forward_distances":
                forward_distances
        }

    # =========================================================
    # 11. XY → 地圖上的 line endpoint
    # =========================================================

    def get_ray_endpoint(
        self,
        ray,
        origin_lat,
        origin_lon
    ):

        start = self.gps_to_xy(

            latitude=ray["latitude"],

            longitude=ray["longitude"],

            origin_lat=origin_lat,

            origin_lon=origin_lon
        )

        direction = np.asarray(
            ray["direction_vector"],
            dtype=float
        )

        norm = np.linalg.norm(
            direction
        )

        if norm == 0:
            return None

        direction = (
            direction / norm
        )

        end = (
            start
            + direction
            * self.ray_length
        )

        lat, lon = self.xy_to_gps(

            x=end[0],

            y=end[1],

            origin_lat=origin_lat,

            origin_lon=origin_lon
        )

        return lat, lon

    # =========================================================
    # 12. 產生互動式地圖
    #
    # Database GPS
    #      ↓
    #      ↓
    #      ↓
    # Estimated Point
    # =========================================================

    def generate_map(
        self,
        rays,
        intersection_result,
        origin_lat,
        origin_lon,
        output_path="outputs/agent3_map.html"
    ):

        if intersection_result is None:

            print(
                "⚠️ 沒有 intersection result"
            )

            return None

        point = intersection_result[
            "point"
        ]

        estimated_lat, estimated_lon = \
            self.xy_to_gps(

                x=point[0],

                y=point[1],

                origin_lat=origin_lat,

                origin_lon=origin_lon
            )

        mean_error = \
            intersection_result.get(
                "mean_error",
                0.0
            )

        max_error = \
            intersection_result.get(
                "max_error",
                0.0
            )

        # -----------------------------------------------------
        # 建立地圖
        # -----------------------------------------------------

        m = folium.Map(

            location=[
                estimated_lat,
                estimated_lon
            ],

            zoom_start=18,

            tiles="OpenStreetMap"
        )

        # -----------------------------------------------------
        # Layer
        # -----------------------------------------------------

        db_layer = folium.FeatureGroup(
            name="Database GPS"
        )

        connection_layer = folium.FeatureGroup(
            name="DB → Estimated Point"
        )

        estimated_layer = folium.FeatureGroup(
            name="Estimated Point"
        )

        db_layer.add_to(m)
        connection_layer.add_to(m)
        estimated_layer.add_to(m)

        # -----------------------------------------------------
        # Estimated Point
        # -----------------------------------------------------

        estimated_popup = f"""
        <div style="font-size:14px">

        <b>⭐ Agent 3 Estimated Point</b>

        <hr>

        <b>Latitude</b><br>
        {estimated_lat:.8f}

        <br><br>

        <b>Longitude</b><br>
        {estimated_lon:.8f}

        <br><br>

        <b>Mean Ray Error</b><br>
        {mean_error:.2f} m

        <br><br>

        <b>Max Ray Error</b><br>
        {max_error:.2f} m

        <br><br>

        <b>Number of Rays</b><br>
        {len(rays)}

        </div>
        """

        folium.Marker(

            location=[
                estimated_lat,
                estimated_lon
            ],

            popup=folium.Popup(
                estimated_popup,
                max_width=350
            ),

            tooltip="⭐ Estimated Point",

            icon=folium.Icon(
                color="red",
                icon="star",
                prefix="fa"
            )

        ).add_to(
            estimated_layer
        )

        # -----------------------------------------------------
        # Database GPS + Connection
        # -----------------------------------------------------

        colors = {

            "east":
                "red",

            "west":
                "blue",

            "north":
                "green",

            "south":
                "purple"
        }

        all_points = [

            [
                estimated_lat,
                estimated_lon
            ]

        ]

        for index, ray in enumerate(rays):

            db_lat = ray[
                "latitude"
            ]

            db_lon = ray[
                "longitude"
            ]

            image = ray.get(
                "image",
                "unknown"
            )

            direction_name = ray.get(
                "direction_name",
                "unknown"
            )

            heading = ray.get(
                "heading",
                0.0
            )

            bearing = ray.get(
                "bearing",
                0.0
            )

            relative_angle = ray.get(
                "relative_angle",
                0.0
            )

            similarity = ray.get(
                "similarity_score",
                0.0
            )

            roi_label = ray.get(
                "roi_label",
                "unknown"
            )

            roi_similarity = ray.get(
                "roi_similarity",
                0.0
            )

            # -------------------------------------------------
            # 顏色
            # -------------------------------------------------

            line_color = colors.get(
                str(direction_name).lower(),
                "blue"
            )

            # -------------------------------------------------
            # DB popup
            # -------------------------------------------------

            popup_html = f"""
            <div style="font-size:14px">

            <b>📷 Database Image {index + 1}</b>

            <hr>

            <b>Image</b><br>
            {os.path.basename(image)}

            <br><br>

            <b>Query Direction</b><br>
            {direction_name}

            <br><br>

            <b>Object</b><br>
            {roi_label}

            <br><br>

            <b>Agent 2 Similarity</b><br>
            {similarity:.4f}

            <br><br>

            <b>ROI Similarity</b><br>
            {roi_similarity:.4f}

            <br><br>

            <b>DB Heading</b><br>
            {heading:.2f}°

            <br><br>

            <b>Object Relative Angle</b><br>
            {relative_angle:.2f}°

            <br><br>

            <b>World Bearing</b><br>
            {bearing:.2f}°

            <br><br>

            <b>GPS</b><br>
            {db_lat:.8f}<br>
            {db_lon:.8f}

            </div>
            """

            # -------------------------------------------------
            # DB Marker
            # -------------------------------------------------

            folium.Marker(

                location=[
                    db_lat,
                    db_lon
                ],

                popup=folium.Popup(
                    popup_html,
                    max_width=350
                ),

                tooltip=(
                    f"DB {index + 1} | "
                    f"{roi_label}"
                ),

                icon=folium.Icon(
                    color="blue",
                    icon="camera",
                    prefix="fa"
                )

            ).add_to(
                db_layer
            )

            # -------------------------------------------------
            # DB → Estimated Point
            # -------------------------------------------------

            line_popup = f"""
            <b>DB → Estimated Point</b>

            <br><br>

            Image:<br>
            {os.path.basename(image)}

            <br><br>

            Object:<br>
            {roi_label}

            <br><br>

            Bearing:<br>
            {bearing:.2f}°

            <br><br>

            Agent 2 Similarity:<br>
            {similarity:.4f}
            """

            folium.PolyLine(

                locations=[

                    [
                        db_lat,
                        db_lon
                    ],

                    [
                        estimated_lat,
                        estimated_lon
                    ]

                ],

                color=line_color,

                weight=4,

                opacity=0.75,

                popup=folium.Popup(
                    line_popup,
                    max_width=300
                ),

                tooltip=(
                    f"{direction_name} "
                    f"→ Estimated Point"
                )

            ).add_to(
                connection_layer
            )

            all_points.append(
                [
                    db_lat,
                    db_lon
                ]
            )

        # -----------------------------------------------------
        # 自動縮放
        # -----------------------------------------------------

        if len(all_points) >= 2:

            m.fit_bounds(
                all_points,
                padding=[
                    30,
                    30
                ]
            )

        # -----------------------------------------------------
        # Layer Control
        # -----------------------------------------------------

        folium.LayerControl().add_to(m)

        # -----------------------------------------------------
        # 儲存
        # -----------------------------------------------------

        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True
        )

        m.save(
            output_path
        )

        print(
            "\n🗺️ Agent 3 Map:"
        )

        print(
            f"   {output_path}"
        )

        print(
            "\n⭐ Estimated GPS:"
        )

        print(
            f"   Latitude  = "
            f"{estimated_lat:.8f}"
        )

        print(
            f"   Longitude = "
            f"{estimated_lon:.8f}"
        )

        print(
            f"\n📏 Mean ray error = "
            f"{mean_error:.2f} m"
        )

        print(
            f"📏 Max ray error = "
            f"{max_error:.2f} m"
        )

        return {

            "map_path":
                output_path,

            "latitude":
                estimated_lat,

            "longitude":
                estimated_lon
        }

    # =========================================================
    # 13. Main localization
    # =========================================================

    def localize(
        self,
        matched_candidates,
        image_sizes,
        output_map="outputs/agent3_map.html"
    ):

        print(
            "\n"
            + "=" * 60
        )

        print(
            "📍 Agent 3 Geometry Localization"
        )

        print(
            "=" * 60
        )

        # -----------------------------------------------------
        # 建立 rays
        # -----------------------------------------------------

        rays = self.build_rays(

            matched_results=
                matched_candidates,

            image_sizes=
                image_sizes
        )

        print(
            f"\n🔎 有效 rays: "
            f"{len(rays)}"
        )

        if len(rays) < 2:

            print(
                "❌ 不足 2 條 ray，"
                "無法進行幾何定位"
            )

            return None

        # -----------------------------------------------------
        # 建立 local coordinate origin
        #
        # 使用所有 DB GPS 平均值
        # -----------------------------------------------------

        origin_lat = float(
            np.mean([
                ray["latitude"]
                for ray in rays
            ])
        )

        origin_lon = float(
            np.mean([
                ray["longitude"]
                for ray in rays
            ])
        )

        # -----------------------------------------------------
        # Ray intersection
        # -----------------------------------------------------

        intersection = \
            self.calculate_ray_intersection(

                rays=rays,

                origin_lat=origin_lat,

                origin_lon=origin_lon
            )

        if intersection is None:

            print(
                "❌ Ray intersection 失敗"
            )

            return None

        # -----------------------------------------------------
        # Estimated GPS
        # -----------------------------------------------------

        point = intersection[
            "point"
        ]

        latitude, longitude = \
            self.xy_to_gps(

                x=point[0],

                y=point[1],

                origin_lat=origin_lat,

                origin_lon=origin_lon
            )

        mean_error = intersection[
            "mean_error"
        ]

        max_error = intersection[
            "max_error"
        ]

        # -----------------------------------------------------
        # Confidence
        #
        # error 越小 → confidence 越高
        # -----------------------------------------------------

        geometry_score = math.exp(
            -mean_error / 20.0
        )

        if rays:

            visual_scores = [

                max(
                    0.0,
                    min(
                        1.0,
                        float(
                            ray.get(
                                "similarity_score",
                                0.0
                            )
                        )
                    )
                )

                for ray in rays

            ]

            visual_score = float(
                np.mean(
                    visual_scores
                )
            )

        else:

            visual_score = 0.0

        confidence = (
            0.5 * geometry_score
            +
            0.5 * visual_score
        )

        # -----------------------------------------------------
        # 產生地圖
        # -----------------------------------------------------

        map_result = self.generate_map(

            rays=rays,

            intersection_result=
                intersection,

            origin_lat=
                origin_lat,

            origin_lon=
                origin_lon,

            output_path=
                output_map
        )

        # -----------------------------------------------------
        # 最終結果
        # -----------------------------------------------------

        result = {

            "latitude":
                latitude,

            "longitude":
                longitude,

            "confidence":
                confidence,

            "mean_intersection_error_m":
                mean_error,

            "max_intersection_error_m":
                max_error,

            "num_rays":
                len(rays),

            "rays":
                rays,

            "map":
                map_result
        }

        # -----------------------------------------------------
        # Terminal output
        # -----------------------------------------------------

        print(
            "\n"
            + "=" * 60
        )

        print(
            "🎯 Agent 3 Final Result"
        )

        print(
            "=" * 60
        )

        print(
            f"Latitude  : "
            f"{latitude:.8f}"
        )

        print(
            f"Longitude : "
            f"{longitude:.8f}"
        )

        print(
            f"Confidence: "
            f"{confidence:.4f}"
        )

        print(
            f"Mean Error: "
            f"{mean_error:.2f} m"
        )

        print(
            f"Max Error : "
            f"{max_error:.2f} m"
        )

        print(
            f"Rays      : "
            f"{len(rays)}"
        )

        print(
            f"Map       : "
            f"{output_map}"
        )

        return result

