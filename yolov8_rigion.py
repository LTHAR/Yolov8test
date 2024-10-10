import cv2
import numpy as np
from ultralytics import YOLO

def get_nearest_chair(image_path):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # 使用 YOLOv8n 预训练模型，您可以根据需要选择不同版本

    # Read the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # Perform inference
    results = model(image)

    # Extract detections
    chairs = []
    for result in results:
        for box in result.boxes:
            if box.cls == 56:  # 56是“chair”的类别索引
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()  # 获取边框的四个坐标
                chairs.append((x1, y1, x2, y2))

    if not chairs:
        print("No chairs detected.")
        return None

    # Calculate image center
    center_x, center_y = width // 2, height // 2

    # Find the nearest chair
    nearest_chair = None
    min_distance = float('inf')

    for (x1, y1, x2, y2) in chairs:
        chair_center_x = (x1 + x2) // 2
        chair_center_y = (y1 + y2) // 2
        distance = np.sqrt((chair_center_x - center_x) ** 2 + (chair_center_y - center_y) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_chair = (x1, y1, x2, y2)

    # Draw the nearest chair on the image
    if nearest_chair:
        x1, y1, x2, y2 = nearest_chair
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 画框
        cv2.putText(image, "Chair", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate corner points
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        print("Nearest chair corners:", corners)

    # Show the image with detected chairs
    cv2.imshow("Detected Chairs", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return nearest_chair

# Example usage
image_path = 'image3.png'  # 替换为您的图像路径
nearest_chair = get_nearest_chair(image_path)
