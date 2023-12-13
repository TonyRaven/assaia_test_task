import os

from object_detector import ObjectDetector


if __name__ == "__main__":

    model = ObjectDetector(os.environ.get("POLYGON_PATH"),
                           os.environ.get("VIDEOS_FOLDER_PATH"))

    model.detect_vehicles_in_videos()

    model.save_predictions(os.environ.get("OUTPUT_PATH"))
