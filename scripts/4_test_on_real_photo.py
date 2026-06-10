from ultralytics import YOLO


for i in range(1, 29):
    model = YOLO("runs/segment/models/yolov8_parts/weights/best.pt")
    results = model(f"test_photos/photo_{i}.jpg", conf=0.1, imgsz=736)
    results[0].save(f"test_photos/result_{i}.jpg")
    #metrics = model.val()
    #print(metrics.box.maps)
    #print(metrics.seg.maps)

    for r in results:
        print("Confidence scores:", r.boxes.conf)
        print("Classes detected:", r.boxes.cls)