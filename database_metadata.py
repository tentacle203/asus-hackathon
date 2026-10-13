# database_metadata.py

import json
import os


class DatabaseMetadata:

    def __init__(self, json_path):

        self.json_path = json_path

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = {}

        # 你的 JSON 格式：
        #
        # {
        #     "node_01_01": {
        #         "node_id": "...",
        #         "image_path": "...",
        #         "heading": ...,
        #         "latitude": ...,
        #         "longitude": ...
        #     }
        # }

        for node_name, item in data.items():

            image_path = item["image_path"]

            # 只取檔名
            image_name = os.path.basename(image_path)

            self.metadata[image_name] = {
                "node_id": item.get("node_id"),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "heading": item.get("heading"),
                "image_path": image_path,
                "image_width": item.get("image_width"),
                "image_height": item.get("image_height")
            }

        print(
            f"📍 [Metadata] 載入 {len(self.metadata)} 筆 GPS metadata"
        )


    def get(self, image_path):

        image_name = os.path.basename(image_path)

        if image_name not in self.metadata:
            return None

        return self.metadata[image_name]


    def get_all(self):

        return self.metadata