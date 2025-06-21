import json
import os
from mission_map_module import MissionMap
from PIL import Image


def prepare_assets(directory: str) -> tuple[str, str]:
    os.makedirs(directory, exist_ok=True)
    map_path = os.path.join(directory, "demo_map.pgm")
    json_path = os.path.join(directory, "demo_map.json")
    img = Image.new("L", (50, 50), 200)
    img.save(map_path)
    info = {
        "resolution": 0.1,
        "origin": [0.0, 0.0, 0.0],
        "width": 50,
        "height": 50,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(info, f)
    return map_path, json_path


def main() -> None:
    asset_dir = "assets"
    output_dir = "demo_output"
    map_pgm, map_json = prepare_assets(asset_dir)

    mm = MissionMap()
    mm.load_map(map_pgm, map_json)
    mm.start()
    for i in range(0, 50, 5):
        mm.update_pose(i * 0.1, i * 0.1, 0.0)
    mm.stop()
    mm.export_results(output_dir)
    print("Demo results saved to", output_dir)


if __name__ == "__main__":
    main()
