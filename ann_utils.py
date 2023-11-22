import os
import cv2
from PIL import Image


def save_point(x,y,id, cfg):
    name = ('00000000' + str(id))[-6:] + '.png'
    path = cfg.data_path + 'annotations.csv'
    
    if not os.path.exists(path):
        with open(path,'w') as f:
            f.write('file_name,image_id,x,y\n')
    
    with open(path,'a') as f:
        f.write("{:},{:},{:},{:}\n".format(name, id,x,y))
        

def save_image(img, id, cfg):
    name = ('00000000' + str(id))[-6:] + '.png'
    path = cfg.data_path + 'images/' + name
    print(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Image.fromarray(img).save(path)
