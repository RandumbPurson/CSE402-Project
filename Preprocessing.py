import os
import sys
import warnings
from types import SimpleNamespace

from PIL import Image

# Add Pytorch_Retinaface/ to path so we can import from it
sys.path.append("Pytorch_Retinaface/")
from detectlib import Detector

def cropFaces(
    src, dest, out_size=(160, 160), regenerate=True,
    args=SimpleNamespace(**{
        "trained_model": 'Pytorch_Retinaface/weights/Resnet50_Final.pth',
        "network": "resnet50",
        "cpu": False,
        "confidence_threshold": 0.02,
        "top_k": 5000,
        "nms_threshold": 0.4,
        "keep_top_k": 750,
        "save_image": False,
        "vis_thres": 0.6
    })
):
    # Check/Generate Directories
    print(f"Checking {dest}")
    if not os.path.isdir(dest):
        os.mkdir(dest)
        print(f"Created destination folder at {os.path.join(os.getcwd(), dest)}")
    
    needed = set(os.listdir(src)) - set(os.listdir(dest))
    
    if len(needed) > 0:
        print(f"Creating class folders: {needed}")
        for cls in needed:
            os.mkdir(os.path.join(dest, cls))
    else:
        n_samples = {cls: len(os.listdir(os.path.join(dest, cls))) for cls in os.listdir(dest)}
        if sum([n == 0 for n in n_samples.values()]) == 0:
            if regenerate == False:
                print(f"Already populated!\n{n_samples}")
                return

    # Crop Images in src dir and save in dest dir
    with warnings.catch_warnings(action="ignore"):
        # ignore UserWarnings
        model = Detector(args)

    classes = os.listdir(src)
    for i, cls in enumerate(classes):
        for img_fname in os.listdir(os.path.join(src, cls)):
            # Load and run model
            img_path = os.path.join(src, cls, img_fname)
            img = Image.open(img_path)
            detections = model.getDetections(img_path)
            if len(detections) < 1:
                print("No detection!")
            xmin, ymin, xmax, ymax, _ = tuple(np.round(detections[0]).astype(int))

            # Crop and save
            cropped = np.array(transforms.functional.resized_crop(
                img, top=ymin, left=xmin, 
                width=xmax - xmin, height=ymax - ymin,
                size=out_size
            ))
            cv2.imwrite(
                os.path.join(dest, cls, img_fname),
                cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)
            )
        print("\r"+" "*80, f"\r[{i+1}/{len(classes)}] Finished cropping class {cls}", end="")
    print("\nFinished cropping!")
    del model
