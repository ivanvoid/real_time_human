import os
import cv2
import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
from types import SimpleNamespace

from ann_utils import *
from model import PoseEstimationNetwork as Net

global data_handle 
data_handle = {
    'mouse_x':0,
    'mouse_y':0,
    'is_new_point_added':False,
    'running_image_id':0,
    'point_counter':0,
}

# train model on collected data


# Mouse labeling callback
def annotation_callback(event, x,y, flags, params):
    global data_handle
    if event == cv2.EVENT_LBUTTONDOWN:
        data_handle['mouse_x'] = x
        data_handle['mouse_y'] = y
        data_handle['is_new_point_added'] = True


def video_stream(cfg, model):
    global data_handle 
    
    cap = cv2.VideoCapture(0)
    
    cv2.namedWindow('Main')
    cv2.moveWindow('Main', 0,0)
    
    # label editing
    cv2.namedWindow('Label_image')
    cv2.moveWindow('Label_image', 640,0)
    saved_frame = np.zeros((480,640,3))
    cv2.setMouseCallback('Label_image', annotation_callback)
    
    # retrain(model, cfg)
    # Load model
    model_name = os.listdir(cfg.model_weight_path)[0]
    m_path = os.path.join(cfg.model_weight_path, model_name)
    model.load_state_dict(torch.load(m_path))

    while True:
        ret, frame = cap.read()
        
        
         # model prediction handling
        with torch.no_grad():
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img).resize(cfg.net_inputs)
            w,h = img.size
            t_image = torch.tensor(np.array(img,dtype=np.int32)).float()
            t_image = t_image.permute(2,0,1)
            out = model(t_image)
            out = out.reshape((4,2))
        new_points = out*cfg.net_inputs[0]
        new_points[0] = new_points[0]/ cfg.net_inputs[0]/w
        new_points[1] = new_points[1]/ cfg.net_inputs[1]/w
        new_points = np.array(new_points).astype(np.int32)
        # print(new_points)
        processed_frame = frame.copy()
        cv2.polylines(processed_frame, [new_points], True,(0,255,0),2)
        
        
        cv2.imshow('Main',processed_frame)
        
        
        # Key handling
        keypress = cv2.waitKey(20)
        if keypress == ord('q'):
            break
        elif keypress == ord('s'):
            saved_frame = frame
            cv2.imshow('Label_image',saved_frame)
            save_image(frame, 
                       data_handle['running_image_id'], 
                       cfg)
        
        # handling addition of annotations
        if (data_handle['is_new_point_added'] and np.mean(saved_frame) > 0.0):
            print(data_handle['point_counter'])
            if data_handle['point_counter'] < 4:
                x = data_handle['mouse_x']
                y = data_handle['mouse_y']
                cv2.circle(saved_frame, (x,y), 5, (0,0,255),-1)
                cv2.imshow('Label_image',saved_frame)
                save_point(x,y,
                           data_handle['running_image_id'],cfg)
                data_handle['point_counter'] += 1
                print(x,y)
            else:
                saved_frame = np.zeros((480,640,3))
                cv2.imshow('Label_image',saved_frame)

                # cv2.destroyWindow('Label_image')
                data_handle['point_counter'] = 0
                data_handle['running_image_id'] += 1
                
                # retrain(model, cfg)
                
            data_handle['is_new_point_added'] = False


       
            
    cap.release()
    cv2.destroyAllWindows()
    


def init_run(cfg):
    im_path = cfg.data_path+'images'
    os.makedirs(im_path, exist_ok=True)
    n = len(os.listdir(im_path))
    global data_handle 
    data_handle['running_image_id'] = n
    
    
if __name__ == '__main__':
    cfg = {
        'data_path':'./data/',
        'model_weight_path':'./model/',
        'net_inputs':(256,256),
        'net_emb_dim':128,
    }
    cfg = SimpleNamespace(**cfg)
    
    model = Net(cfg)
    
    init_run(cfg)
    
    video_stream(cfg, model)


print('done.')
