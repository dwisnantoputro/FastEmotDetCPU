from __future__ import print_function
import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time

import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg
from layers.functions.prior_box import PriorBox
#from utils.nms_wrapper import nms
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.model import Face
from utils.box_utils import decode
from utils.timer import Timer
from IPython.display import Image
from matplotlib import pyplot as plt
#from torchsummary import summary


from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

from utils2 import get_initial_weights
from layers2 import BilinearInterpolation
from keras_layer_normalization import LayerNormalization
import tensorflow as tf

det_time = 0
det_time_tot = 0
class_time = 0
class_time_tot = 0
n = 0

emotion_model_path = 'models/kdef_best.hdf5'

emotion_classifier = load_model(emotion_model_path, compile=False, custom_objects={'BilinearInterpolation': BilinearInterpolation, 'LayerNormalization':LayerNormalization, 'tf':tf})


EMOTIONS = ["Angry" ,"Disgust","Fear", "Happy", "Neutral", "Sad", "Surprised"]

parser = argparse.ArgumentParser(description='Face')

parser.add_argument('-m', '--trained_model', default='weights/Final_LWFCPU.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('--nms_threshold', default=0.3, type=float, help='nms_threshold')
parser.add_argument('-r', '--record', action="store_true", default=False, help='record video')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--frames', default=1000, type=int, help='number of frames')

args = parser.parse_args()

def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
			
def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

resize = 1
numframes = args.frames
fps = 20

vc = cv2.VideoCapture(0)
frame_width = int(vc.get(3))
frame_height = int(vc.get(4))
   
if __name__ == '__main__':
    torch.set_grad_enabled(False)
    # net and model
    net = Face(phase='test', size=None, num_classes=2)    # initialize detector
    net = load_model(net, args.trained_model, True)
    net.eval()
    device = torch.device("cpu")
    net = net.to(device) 
    #summary(net, (3, 1024, 1024))

    _t = {'forward_pass': Timer(), 'misc': Timer()}

    while 1:
        n = n + 1
        rval, img_raw = vc.read()

        _t['forward_pass'].tic()

        gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)   
        RGB = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)  
        canvas = np.zeros((250, 300, 3), dtype="uint8")
        canvas_full = np.zeros((250, 1000, 3), dtype="uint8")
        frameClone = img_raw.copy() 
   
        img = np.float32(img_raw)

        
        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
        
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        _t['forward_pass'].tic()
        out = net(img)  # forward pass
        _t['forward_pass'].toc()

        priorbox = PriorBox(cfg, out[2], (im_height, im_width), phase='test')
        priors = priorbox.forward()
        priors = priors.to(device)
        loc, conf, _ = out
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > 0.01)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        #keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:750, :]
        
        det_time = _t['forward_pass'].average_time
        det_time_tot = det_time_tot + det_time
        fps_det = "fps_det = {:.2f}".format(1/det_time)
        cv2.putText(img_raw, fps_det, (10, 420),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))


        num_face =0
        nf = 0
        for b in dets:
            if b[4] < args.vis_thres:
                continue
            
            num_face = num_face+1

            text = "{:.4f}".format(b[4])
            b = list(map(int, b))

            #ROI
            fX = b[0]
            fY = b[1]
            fW = b[2]
            fH = b[3]
            roi = gray[fY:fH, fX:fW]

            
            if roi.shape[0]!=0 and roi.shape[1]!=0:
                roi_s = cv2.resize(roi, (48, 48), cv2.INTER_LINEAR)
            
            _t['misc'].tic()

            roi = roi_s.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)       
            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]

            _t['misc'].toc()
            class_time = _t['misc'].average_time
            class_time_tot = class_time_tot + class_time
            fps_class = "fps_class = {:.2f}".format(1/class_time)
            cv2.putText(img_raw, fps_class, (10, 440),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)) 

            fps_tot = "fps_tot = {:.2f}".format(1/(class_time+det_time)) 
            cv2.putText(img_raw, fps_tot, (10, 460),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255)) 


            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                cv2.putText(img_raw, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
                cv2.rectangle(img_raw, (fX, fY), (fW, fH),
                              (255, 200, 0), 1)

        cv2.imshow("preview", img_raw)
        cv2.destroyWindow("Probabilities-{}".format(num_face+1))
        cv2.destroyWindow("ROI-{}".format(num_face+1))
        
  
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")
    
total_ave_time = (det_time_tot+class_time_tot)/(n)
print('frame_total: {:d} '.format(n+1))
print('time_tot: {:.4f}s'.format(det_time_tot+class_time_tot))   
print('time_ave: {:.4f}s'.format(total_ave_time))
print('fps: {:.4f}'.format(1/(total_ave_time)))
