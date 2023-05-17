from typing import Any
import cv2
import numpy as np
import mediapipe as mp
import json
import os
from pathlib import Path
from ultralytics import YOLO

class MyJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super(MyJSONEncoder, self).default(o)

class HandDetection:
    def __init__(self, file_path, save_path, image, file, draw_person_rect: bool = False, draw_hand_rect: bool = False):
        '''
        file_path: 파일 경로
        save_path: 결과 저장 경로
        image: 이미지
        file: 이미지 파일 이름(확장자 포함)
        draw_person_rect: 사람 영역 표시 여부
        draw_hand_rect: 손 영역 표시 여부
        '''
        self.file_path = file_path
        self.save_path = save_path
        self.image = image
        self.file = file
        self.file_name = file.split('.')[0]
        self.draw_person_rect = draw_person_rect
        self.draw_hand_rect = draw_hand_rect

        self.classes = []
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.person_pos = []
        self.hand_ranges = []

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        self.__find_object()

    # Yolo로 물체 찾기
    def __find_object(self):
        # Load YOLO
        model = YOLO('yolov8x.pt')

        self.height, self.width, self.channels = self.image.shape

        results = model.predict(source=self.image)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)
        confs = results[0].boxes.conf.cpu().numpy().astype(float)

        for box, cls, conf in zip(boxes, classes, confs):
            if cls == 0 and conf > 0.5:
                # center_x, center_y, w, h = box
                # self.boxes.append(center_x, center_y, w, h)
                self.boxes.append([box[0], box[1], box[2] - box[0], box[3] - box[1]])
                self.confidences.append(conf)

        # # Load Yolo
        # net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        
        # with open("coco.names", "r") as f:
        #     self.classes = [line.strip() for line in f.readlines()]
        # layer_names = net.getLayerNames()
        # output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        # self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # self.height, self.width, self.channels = self.image.shape

        # blob = cv2.dnn.blobFromImage(self.image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        # net.setInput(blob)
        # outs = net.forward(output_layers)

        # for out in outs:
        #     for detection in out:
        #         scores = detection[5:]
        #         class_id = np.argmax(scores)
        #         confidence = scores[class_id]
        #         if confidence > 0.5:
        #             # Object detected
        #             center_x = int(detection[0] * self.width)
        #             center_y = int(detection[1] * self.height)
        #             w = int(detection[2] * self.width)
        #             h = int(detection[3] * self.height)
        #             # Rectangle coordinates
        #             x = int(center_x - w / 2)
        #             y = int(center_y - h / 2)
        #             self.boxes.append([x, y, w, h])
        #             self.confidences.append(float(confidence))
        #             self.class_ids.append(class_id)

        self.__find_person(self.draw_person_rect)

    def __find_person(self, draw_rect = False):
        indexes = cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN

        for i in range(len(self.boxes)):
            if i in indexes:
                x, y, w, h = self.boxes[i]
                # label = str(self.classes[self.class_ids[i]])
            # if label == "person":
                if draw_rect:
                    # color = self.colors[i]
                    color = (0, 0, 255)
                    cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                    # cv2.putText(self.image, label, (x, y + 30), font, 3, color, 3)
                    cv2.putText(self.image, "person", (x, y + 30), font, 3, color, 3)

                self.person_pos.append([x, y, x + w, y + h])

                ny = y - 50
                nx = x - 50
                y_1 = ny if ny >= 0 else 0
                x_1 = nx if nx >= 0 else 0

                ny = y + h + 50
                nx = x + w + 50
                y_2 = ny if ny <= self.height else self.height
                x_2 = nx if nx <= self.width else self.width

                tmp_image = self.image[y_1:y_2, x_1:x_2]
                if len(tmp_image) == 0:
                    continue
                self.__find_hand(tmp_image, [x_1, y_1, x_2, y_2], self.draw_hand_rect)

        self.__save_result()

    # 손위치 찾기(mediapipe 이용)
    def __find_hand(self, person_image, person_rect: list, draw_hand_rect = False):
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=10,
            min_detection_confidence=0.35
            ) as hands:
            results = hands.process(cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB))
            
            if not results.multi_hand_landmarks:
                return
            image_height, image_width, _ = person_image.shape
            annotated_image = person_image.copy()
            
            all_x, all_y = [], []
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style()
                )

                for hnd in self.mp_hands.HandLandmark:
                    all_x.append(int(hand_landmarks.landmark[hnd].x * image_width)) # multiply x by image width
                    all_y.append(int(hand_landmarks.landmark[hnd].y * image_height)) # multiply y by image height

            pos = [min(all_x), min(all_y), max(all_x), max(all_y)]

            nhx = pos[0] - 30
            nhy = pos[1] - 30
            hx_1 = nhx if nhx >= 0 else 0
            hy_1 = nhy if nhy >= 0 else 0

            nhx = pos[2] + 30
            nhy = pos[3] + 30
            hx_2 = nhx if nhx <= image_width else image_width
            hy_2 = nhy if nhy <= image_height else image_height

            x_1 = person_rect[0]
            y_1 = person_rect[1]

            if draw_hand_rect:
                cv2.rectangle(img=self.image, pt1=(x_1 + hx_1, y_1 + hy_1), pt2=(x_1 + hx_2, y_1 + hy_2), color=(36,255,12), thickness=1)
            self.hand_ranges.append([x_1 + hx_1, y_1 + hy_1, x_1 + hx_2, y_1 + hy_2])

    # 손 위치 json 저장
    def __save_result(self):
        jsonfile_name = self.file_name + '.json'
        if os.path.isfile(self.save_path + '/' + jsonfile_name):
            with open(self.save_path + '/' + jsonfile_name, 'r') as file:
                input_data = file.read()
                output_dict = json.loads(input_data)
        else:
            output_dict = []

        output_image_dict = {
            "image": self.file,
            "verified": False,
            "annotations": []
        }

        for point in self.hand_ranges:
            x1 = point[0]
            x2 = point[2]
            y1 = point[1]
            y2 = point[3]

            width = x2 - x1
            height = y2 - y1
            x = x1 + width / 2
            y = y1 + height / 2

            shape_dict = {
                "label": "Hands",
                "coordinates": {
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height
                }
            }

            output_image_dict["annotations"].append(shape_dict)
        
        exists = False
        for i in range(0, len(output_dict)):
            if output_dict[i]["image"] == output_image_dict["image"]:
                exists = True
                output_dict[i] = output_image_dict
                break
        
        if not exists:
            output_dict.append(output_image_dict)
        
        cv2.imwrite(self.save_path + '/' + self.file, self.image)
        Path(self.save_path + '/' + jsonfile_name).write_text(json.dumps(output_dict, cls=MyJSONEncoder))


def main(file_path, save_path):
    file_list = os.listdir(file_path)
    for file in file_list:
        image = cv2.imread(file_path + '/' + file)
        HandDetection(file_path, save_path, image, file, True, True)

if __name__ == '__main__':
    file_path = './origin'
    save_path = './result'

    main(file_path, save_path)