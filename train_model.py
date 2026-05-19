from ultralytics import YOLO
from multiprocessing import freeze_support


def main():

    print("Starting YOLOv8 Training...")

    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="data.yaml",
        epochs=20,
        imgsz=640,
        batch=4,
        workers=0,
        project="runs",
        name="traffic_model_final"
    )

    print("\nTraining Completed!")
    print("Model Saved Successfully")


if __name__ == "__main__":
    freeze_support()
    main()
