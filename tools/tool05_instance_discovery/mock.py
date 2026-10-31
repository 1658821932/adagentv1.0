"""Local mock implementation of the instance discovery tool."""


def run(input_data: dict) -> dict:
    """Return mock traffic-participant detections."""
    return {
        "status": "success",
        "tool": "instance_discovery",
        "result": {
            "instance_info_path": input_data["output_path"],
            "instances": [
                {
                    "instance_id": 0,
                    "class": "car",
                    "visible_frames": [0, 100],
                    "bbox": [620, 280, 760, 430],
                    "mask_path": "outputs/mock/masks/vehicle_01.png",
                },
                {
                    "instance_id": 1,
                    "class": "car",
                    "visible_frames": [0, 100],
                    "bbox": [320, 300, 430, 420],
                    "mask_path": "outputs/mock/masks/vehicle_02.png",
                },
            ]
        },
    }
