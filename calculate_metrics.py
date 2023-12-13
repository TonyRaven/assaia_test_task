import json
import os


def calculate_metrics_(model_predictions_path, real_time_intervals_path):
    with open(model_predictions_path, 'r') as f:
        dict_a = json.load(f)
    with open(real_time_intervals_path, 'r') as f:
        dict_b = json.load(f)

    precision = {}
    recall = {}
    f1 = {}
    for video_id, a in dict_a.items():
        result = []
        b = dict_b[video_id]
        for range_a in a:
            for range_b in b:
                if range_a[1] >= range_b[0] and range_a[0] <= range_b[1]:
                    intersection_start = max(range_a[0], range_b[0])
                    intersection_end = min(range_a[1], range_b[1])

                    result.extend(range(intersection_start,
                                        intersection_end + 1))
        intersection = len(result)
        amount_a = sum([(range_a[1] - range_a[0] + 1) for range_a in a])
        amount_b = sum([(range_b[1] - range_b[0] + 1) for range_b in b])
        precision[video_id] = round(intersection / amount_a, 2)
        recall[video_id] = round(intersection / (amount_b + 10 ** -10), 2)
        f1[video_id] = round(2 * precision[video_id] * recall[video_id] / (
                    precision[video_id] + recall[video_id] + 10 ** -10), 2)
    return {'Precision': precision, 'Recall': recall, 'F1': f1}


if __name__ == "__main__":
    metrics = calculate_metrics_(os.environ.get("PREDICTED_INTERVALS_PATH"),
                                 os.environ.get("REAL_INTERVALS_PATH"))
    print(metrics)
