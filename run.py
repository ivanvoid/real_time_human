import cv2
import numpy as np
import torch 
import torch.nn as nn 

global data_handle 
data_handle = {
    'mouse_x':0,
    'mouse_y':0,
    'is_new_point_added':False
}

# collect(annotate) data
# train model on collected data

# Mouse labeling callback
def annotate(event, x,y, flags, params):
    global data_handle
    if event == cv2.EVENT_LBUTTONDOWN:
        data_handle['mouse_x'] = x
        data_handle['mouse_y'] = y
        data_handle['is_new_point_added'] = True

class PoseEstimationNetwork(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        print(x.shape)
        return x


def video_stream():
    global data_handle 
    
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow('Main')
    
    # label editing
    cv2.namedWindow('Label_image')
    saved_frame = np.zeros((480,640,3))
    cv2.setMouseCallback('Label_image', annotate)
    
    while True:
        ret, frame = cap.read()
        cv2.imshow('Main',frame)
        
        keypress = cv2.waitKey(20)
        if keypress == ord('q'):
            break
        elif keypress == ord('s'):
            saved_frame = frame
            cv2.imshow('Label_image',saved_frame)
        
        if data_handle['is_new_point_added']:
            x = data_handle['mouse_x']
            y = data_handle['mouse_y']
            cv2.circle(saved_frame, (x,y), 5, (0,0,255),-1)
            cv2.imshow('Label_image',saved_frame)
            data_handle['is_new_point_added'] = False

    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == '__main__':
    model = PoseEstimationNetwork()
    
    video_stream()



print('done.')