# Task 1: Implement an object detector which identifies the classes of the objects in an image or video.
# >> Kiran L. Ware

# Here we import some required Library Opencv (cv2)
import cv2

# set threshold value to detect objects.
thres = 0.65

# use for capturing or accessing video which we select for task.
cap = cv2.VideoCapture('road_trafifc.mp4')
# image = cv2.imread('task1.jpg')

cap.set(3, 640)
cap.set(4, 480)

# Here we have one name file saved as a name coco.names.
# classNames = []
classFile = 'coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    ret, img = cap.read()

    classIds, conf, bbox = net.detect(img, confThreshold=thres)
    print(classIds, bbox)
    if len(classIds) != 0:
        for classId, conf, box in zip(classIds.flatten(), conf.flatten(), bbox):
            # creat rectangle around detecting image. (Yellow Color Frame)
            cv2.rectangle(img, box, color=(0, 255, 255), thickness=2)
            # text of detecting image (Red in color)
            cv2.putText(img, classNames[classId-1], (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            # for confidence detection (now we disabled confidence for a while)
            cv2.putText(img, str(round(conf*100, 2)), (box[0] + 350, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0,
                        (0, 255, 0), 0)

    cv2.imshow("Output", img)
    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
print('Thank You..!')