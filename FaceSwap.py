import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt

#import matplotlib.image as mpimg

import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

print('insightface', insightface.__version__)
print('numpy', np.__version__)

# Initialize FaceAnalysis app

#Detect faces
# Initialize FaceAnalysis app with age and gender prediction enabled
app = FaceAnalysis(det_batchify=True, det_extra=True, age=True, gender=True)
app.prepare(ctx_id=0,det_size=(640,640))


swapper = insightface.model_zoo.get_model('inswapper_128.onnx',
                                          download = False,
                                          download_zip = False
                                          )

# Load an image which is the target
destination = cv2.imread('Munja_Group.jpg')
# Convert BGR to RGB
destination = cv2.cvtColor(destination, cv2.COLOR_BGR2RGB)

plt.imshow(destination)
plt.axis('off')
plt.show()

faces = app.get(destination)

print("\n\n *** Number of faces in destination image are: " + str(len(faces)) + "\n\n")

# Load an image which is the source
#source = cv2.imread('Ridhima1.jpg')
# Convert BGR to RGB

#source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

#plt.imshow(source)
#plt.axis('off')
#plt.show()

#Detect the face

#source_faces = app.get(source)

#print("\n\n *** Number of faces in source image are: " + str(len(source_faces)) + "\n\n")

#source_face = source_faces[0]

#Replace faces in friends image
res = destination.copy()
# convert bgr to rgb 
#res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
counter = 0

for face in faces:
    counter = counter + 1
    #if counter == 0:
        # Extract face bounding box coordinates
    bbox = face['bbox']
    x, y, w, h = [int(coord) for coord in bbox]

    # Crop face from the image
    face_img = res[y:y+h, x:x+w]

    # Predict age and gender
    age, gender = face['age'], face['gender']

    # Print age and gender
    #print(f"\n\n\nImage " + str(counter) + " --> Predicted Age: {age}, Predicted Gender: {gender}")
    print("\n\nImage " + str(counter) + " :")
    print(f"Predicted Age: {age}, Predicted Gender: {gender}")

    # Draw bounding box and text on the original image
    cv2.rectangle(res, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(res, f"Age: {age}, Gender: {gender}", (x, y - 10),
    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    #    res = swapper.get(res, face, source_face, paste_back=True)
    #    counter = counter+1
    #else:
    #    counter = counter+1
    #    print(counter)

#Replace first face with External image

# convert bgr to rgb 
#res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

plt.imshow(res)
#plt.imshow(res[:,:,::-1])

plt.axis('off')
plt.show()

