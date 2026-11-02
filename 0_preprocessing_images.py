import os
import cv2
import numpy as np

### Configure directories dynamically
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
INPUT_DIR = os.path.join(BASE_DIR, "data", "raw_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "resized_images", "images") # the extra images folder is needed for the later scripts
os.makedirs(OUTPUT_DIR, exist_ok=True)


### set size and sensitivity of the object detection
TARGET = 224           # height and width of final image in px
INNER = 185            # target max object height/width in px 
dilate_iterations = 5  # liberal object detection

### mask defintion for object detection
def make_mask(img_bgr): 
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # find foreground of image dynamically in case of off-whites
    mask = cv2.adaptiveThreshold(255-gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 31, 5) 
        
    #find object 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    return mask

def process_image(path_in, path_out):
    img = cv2.imread(path_in, cv2.IMREAD_COLOR)

    h_img, w_img = img.shape[:2] # return height and width

    mask = make_mask(img) # apply mask
    pts = cv2.findNonZero(mask)

    #if not found: take original image, resize, add white border
    if pts is None:                     
        canvas = np.ones((TARGET, TARGET, 3), dtype=np.uint8) * 255
        # scale to fit inner variable if large
        scale = min(1.0, INNER / max(w_img, h_img))
        new_w = max(1, int(w_img * scale))
        new_h = max(1, int(h_img * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        x_off = (TARGET - new_w) // 2
        y_off = (TARGET - new_h) // 2
        canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
        cv2.imwrite(path_out, canvas)
        return True

    # crop out object of image
    x, y, w, h = cv2.boundingRect(pts) 
    crop = img[y:y+h, x:x+w]              # crop to bounding rectangle

    # scale so the largest side equals the inner variable 
    max_side = max(w, h)
    scale = INNER / max_side
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # place centered on 224 x 224 
    canvas = np.ones((TARGET, TARGET, 3), dtype=np.uint8) * 255
    x_off = (TARGET - new_w) // 2
    y_off = (TARGET - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    cv2.imwrite(path_out, canvas)
    return True

# run
for fname in os.listdir(INPUT_DIR):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')): 
        continue
    in_path = os.path.join(INPUT_DIR, fname)
    out_path = os.path.join(OUTPUT_DIR, fname)
    ok = process_image(in_path, out_path)

print("Done", OUTPUT_DIR)
