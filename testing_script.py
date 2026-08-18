import os
import cv2
import numpy as np

split = "test"
clip = "actioncliptest00035"
stem = "actioncliptest00035_36"

grey_path = os.path.join("data/RVSOD", split, "ranking saliency masks/img", clip, stem + ".png")
img_path = os.path.join("data/RVSOD", split, "img", clip, stem + ".jpg")

grey = cv2.imread(grey_path, cv2.IMREAD_GRAYSCALE)
img = cv2.imread(img_path)

objs = []
for level in sorted(int(v) for v in np.unique(grey) if v > 0):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((grey == level).astype(np.uint8), 8)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        comp = (labels == i).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        polygons = [c.reshape(-1).tolist() for c in contours if c.shape[0] >= 3]
        objs.append({"bbox": (x, y, w, h), "polygons": polygons, "level": level})

objs.sort(key=lambda o: o["level"], reverse=True)
for r, o in enumerate(objs, 1):
    o["rank"] = r
    print(r, o["bbox"], o["level"], "npoly=", len(o["polygons"]))

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]
vis = img.copy()
for o in objs:
    c = colors[(o["rank"] - 1) % len(colors)]
    x, y, w, h = o["bbox"]
    cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
    for poly in o["polygons"]:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], True, c, 2)
    cv2.putText(vis, f"rank {o['rank']}", (x, max(y - 5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)

cv2.imwrite("testing_script_vis.jpg", vis)
print("wrote testing_script_vis.jpg")
