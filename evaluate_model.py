from ultralytics import YOLO
import os

# Model path
MODEL_PATH = r"C:\Users\ASUS\runs\detect\runs\traffic_model_final7\weights\best.pt"


def main():

    print("=== MODEL EVALUATION ===")

    if not os.path.exists(MODEL_PATH):
        print("Model not found!")
        return

    # Load model
    model = YOLO(MODEL_PATH)

    # Validate model
    results = model.val(data="data.yaml", split="val")

    # Metrics
    accuracy = results.box.map50 * 100
    precision = results.box.mp * 100
    recall = results.box.mr * 100

    print("\n=== PERFORMANCE REPORT ===")

    print(f"Accuracy : {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall   : {recall:.2f}%")


if __name__ == "__main__":
    main()

    
