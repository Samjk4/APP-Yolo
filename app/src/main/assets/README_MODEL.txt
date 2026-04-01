把 YOLOv8 的 ONNX 模型放在這裡，檔名請用：

  yolov8n.onnx

建議使用 Ultralytics YOLOv8n 的 COCO 模型（80 類），本專案會把：
- person 視為「人」
- car/bus/truck/motorcycle/bicycle 視為「車」
- 其他 COCO 類別統一視為「障礙物」

如果你想改成自訓練的「人/車/障礙物」三分類模型，也可以；
但輸出格式需與 YOLOv8 ONNX export 相容（常見為 [1, 84, 8400]）。

