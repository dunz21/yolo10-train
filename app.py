import cv2
import tempfile
from ultralytics import YOLOv10
from datetime import datetime


def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    # model = YOLOv10.from_pretrained(f'jameslahm/{model_id}')
    model = YOLOv10()
    model.load(weights='runs/detect/train12/weights/best.pt')
    if image:
        results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
        annotated_image = results[0].plot()
        return annotated_image[:, :, ::-1], None
    else:
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_video_path = tempfile.mktemp(suffix=".webm")
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        return None, output_video_path

# if __name__ == '__main__':
#     models = choices=[
#                         "yolov10n",
#                         "yolov10s",
#                         "yolov10m",
#                         "yolov10b",
#                         "yolov10l",
#                         "yolov10x",
#                     ]
#     video_path = '/home/diego/Documents/MivoRepos/mivo-project/footage-apumanque/apumanque_entrada_2_20240705_0900_short_condensed.mkv'
#     image, video, model_id, image_size, conf_threshold = None, video_path, "yolov10m", 640, 0.25
#     # image = '/home/diego/Pictures/2024-07-15_11-47_1.png'
#     annotated_image, output_video_path = yolov10_inference(image, video, model_id, image_size, conf_threshold)
#     if image is not None:
#         cv2.imshow("Annotated Image", image)
#         cv2.waitKey(0)
#     else:
#         cap = cv2.VideoCapture(output_video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             cv2.imshow("Annotated Video", frame)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
#         cap.release()
#     cv2.destroyAllWindows()
    

# if __name__ == '__main__':
#         from ultralytics import YOLO, checks, hub
#     checks()

#     hub.login('39de99922d377cebc47a5b068afb5d0578cdb2bb5d')

#     model = YOLO('https://hub.ultralytics.com/models/BBwERMA5fiOdjYfyT1NW')
#     results = model.train()
    
    
if __name__ == '__main__':
    model = YOLOv10()
    # If you want to finetune the model with pretrained weights, you could load the 
    # pretrained weights like below
    # model = YOLOv10.from_pretrained('jameslahm/yolov10s')
    # or
    # wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
    WEIGHTS = 'yolov10m.pt'
    model = YOLOv10(WEIGHTS)
    dataset = '/home/diego/Documents/MivoRepos/yolov10/datasets/Lorenzo di Pontti Tobalaba.v1i.yolov9/data.yaml'
    project_name = f"{dataset.split('/')[-2]}_{WEIGHTS.split('.')[0]}_{datetime.now().strftime('%H_%M_%S')}"
    model.train(data=dataset, epochs=250, batch=16, imgsz=640, freeze=[0,1,2] , lr0=0.01, device='0', project='runs/train', name=project_name, exist_ok=True, model=WEIGHTS)
    