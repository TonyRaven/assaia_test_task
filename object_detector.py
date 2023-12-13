import json
import os
from collections import defaultdict
from typing import List

import cv2
import torch
from PIL import Image
from shapely.geometry import Polygon, box
from transformers import DetrImageProcessor, DetrForObjectDetection

from config import vehicle_classes


class ObjectDetector:
    def __init__(self, polygons_path: str, videos_folder_path: str) -> None:
        self.processor = DetrImageProcessor.\
            from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.\
            from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.polygons = self._load_polygons_data(polygons_path)
        self.videos_folder_path = videos_folder_path
        self.vehicle_detected_frames = defaultdict(list)

    def _load_polygons_data(self, polygons_path: str) -> List[List[int]]:
        with open(polygons_path, 'r') as file:
            polygons_data = json.load(file)
        return polygons_data

    def _load_time_intervals_data(self,
                                  time_intervals_path: str) -> List[List[int]]:
        with open(time_intervals_path, 'r') as file:
            time_intervals_data = json.load(file)
        return time_intervals_data

    def detect_vehicles_in_videos(self) -> None:
        video_ids = [filename for filename in
                     os.listdir(self.videos_folder_path)
                     if filename.endswith('.mp4')]
        for video_id in video_ids:
            print(f'Видео {video_id} в обработке')
            self._process_video(video_id)

    def _process_video(self, video_id: str) -> None:
        polygon = self.polygons[video_id]
        video_path = self.videos_folder_path + '\\' + video_id
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)

            results = self._detect_objects(pil_image)

            for score, label, box_ in zip(results["scores"], results["labels"],
                                          results["boxes"]):
                object_type = self.model.config.id2label[label.item()]
                if object_type in vehicle_classes:
                    box_ = [round(i) for i in box_.tolist()]
                    if ObjectDetector._is_intersection(box_, polygon):
                        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        self.vehicle_detected_frames[video_id].append(
                            frame_number)

    def _detect_objects(self, pil_image: Image.Image) -> dict:
        inputs = self.processor(images=pil_image, return_tensors="pt")
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]])
        results = self.processor.\
            post_process_object_detection(outputs,
                                          target_sizes=target_sizes,
                                          threshold=0.8)[0]
        return results

    @staticmethod
    def split_list_on_difference(lst: List[int]) -> List[List[int]]:
        result = []
        current_sublist = [lst[0]]

        for i in range(1, len(lst)):
            if abs(lst[i] - lst[i - 1]) > 1:
                result.append([current_sublist[0], current_sublist[-1]])
                current_sublist = [lst[i]]
            else:
                current_sublist.append(lst[i])

        if current_sublist:
            result.append([current_sublist[0], current_sublist[-1]])

        return result

    def save_predictions(self, output_path: str) -> None:
        predictions = {}
        for video_id, frames in self.vehicle_detected_frames.items():
            predictions[video_id] = ObjectDetector.split_list_on_difference(
                frames)
        with open(output_path, 'w') as file:
            json.dump(predictions, file)

    @staticmethod
    def _is_intersection(rectangle_coords: List[int],
                         polygon_coords: List[List[int]]) -> bool:

        rectangle = box(rectangle_coords[0], rectangle_coords[1],
                        rectangle_coords[2], rectangle_coords[3])
        polygon = Polygon(polygon_coords)

        return rectangle.intersects(polygon)

    def calculate_metrics(self, times_intervals, model_answers):
        precision = {}
        recall = {}
        f1 = {}
        for video_id, pred_frames in model_answers.items():
            pred_frames = list(set(pred_frames))
            intersection = 0
            for sub_t in times_intervals[video_id]:
                low, high = sub_t[0], sub_t[1]
                for pred_frame in pred_frames:
                    if pred_frame >= low and pred_frame <= high:
                        intersection += 1
            amount_t = sum([(sub_t[1] - sub_t[0] + 1) for sub_t in
                            times_intervals[video_id]])
            precision[video_id] = intersection / len(pred_frames)
            recall[video_id] = intersection / (amount_t + 10 ** -10)
            f1[video_id] = 2 * precision[video_id] * recall[video_id] / (
                        precision[video_id] + recall[video_id])
        return {'Precision': precision, 'Recall': recall, 'F1': f1}

    def save_metrics(self, metrics_path):
        metrics = self.calculate_metrics(self.time_intervals,
                                         self.vehicle_detected_frames)
        with open(metrics_path, "w") as outfile:
            json.dump(metrics, outfile)

