# -*- coding: utf-8 -*-
"""
created 28/7
@author: Lucid
"""
import openslide 
import os, sys
# from pathlib import Path
import xml.etree.ElementTree as ET
# import openslide

# from openslide.deepzoom import DeepZoomGenerator                    
import numpy as np   
import torch
# import yaml
from tqdm import tqdm
import time
from models.experimental import attempt_load
from utils.datasets import create_dataloader_custom, letterbox
from utils.general import check_img_size, box_iou, non_max_suppression,\
    scale_coords, xyxy2xywh, xywh2xyxy, set_logging, colorstr

from utils.torch_utils import select_device, TracedModel

from inference_utils import utils
# from multiprocessing import Pool
import concurrent.futures


class inference():    
    def __init__(self,dump_path,batch_size=8):
        self.device = select_device('0')    
        self.weights = 'best.pt'
        self.img_size = 640
        self.conf_thres = 0.5
        self.iou_thres = 0.5
        self.classes = None
        self.nm_p =221
        self.no_trace = True
        self.half = False
        self.model = self.load_model()
        self.batch_size = batch_size        
        

    def load_model(self):
        # try:
        # set_logging()            
        self.half = (self.device != 'cpu')  # half precision only supported on CUDA        
        # print(self.weights)
        # print(self.device)
        self.model = attempt_load(self.weights,map_location=self.device)  # 
        print('model_loaded')
        
        if not self.no_trace:
            self.model = TracedModel(self.model, self.device, self.img_size)        
        if self.half:
            self.model.half()  # to FP16
            print('half')
        return self.model
        # except Exception as e:
        #     print(e)           

   
    def fetch_batch(self,batch):  
        lefts,tops,shapes,imgs= [],[],[],[]
        # print('batch length',len(batch))
        for left,top,shape,img in batch:            
            lefts.append(left)
            tops.append(top)                    
            shapes.append(shape)
            imgs.append(img)            
        imges = np.concatenate(imgs.copy()) 
        return lefts.copy(),tops.copy(),shapes.copy(),imges                

    def run_batch_inference(self,batch,anote,wsi_path):                        
        lefts, tops, shapes, img = self.fetch_batch(batch)        
        img = torch.from_numpy(img).to(self.device)                  
        img = img.half() if self.half else img.float()  # uint8 to fp16/32        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)                
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak                        
            pred = self.model(img, augment=False)[0]        
        # Apply NMS
        # print(img.shape[2:])
        pred = non_max_suppression(pred,self.conf_thres,self.iou_thres, self.classes,False)              
        for i, det in enumerate(pred):  # detections per image                            
            left, top = lefts[i],tops[i]            
            gn = torch.tensor(shapes[i])[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                #det[:, :4] = scale_coords((img.shape[2:], det[:, :4],(shapes[i][1],shapes[i][2])).round()
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4],shapes[i]).round()
                #print(img.shape[2:],shapes[i],coords[i])    
                for *xyxy, conf, cls in reversed(det): 					                    
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh                        
                    line = ([*xywh])  #label format                                        
                    #print(line)
                    x1 = left + (float(line[0]) - float(line[2])/2)*(shapes[i][0])
                    y1 = top + (float(line[1]) - float(line[3])/2)*(shapes[i][1])
                    x2 = x1 + float(line[2])*shapes[i][0]
                    y2 = y1 + float(line[3])*shapes[i][1]
                    anote.append([x1,y1,x2,y2,round(float(conf),3),wsi_path])                        
        return anote    
    

    def process(self,wsi_path,pred,tile_size=1024):
        # lefts,tops,imgs = [],[],[]
        x1,y1,x2,y2,conf,blah = pred
        cx = int((x1+x2)/2) 
        cy = int((y1+y2)/2) 
        #breath in pixels        
        breath = abs(x2-x1)
        length = abs(y2-y1)           
        #centering the Groundtruth        
        xc, yc = tile_size/2, tile_size/2    
        left = int(cx -xc- breath/2)               
        top = int(cy -yc- length/2)                                   
        slide = openslide.open_slide(wsi_path)    
        level = slide.get_best_level_for_downsample(1.0 / 40)        
        tile = slide.read_region((left,top),level,(tile_size,tile_size))
        slide.close()
        im = tile.convert('RGB')    
        img0 = np.array(im)    
        shape = (img0.shape[0],img0.shape[1])
        img = letterbox(img0,auto=False)[0]           
        img = img.transpose(2,0,1)
        img = np.expand_dims(img, 0) 
        return left,top,shape,img
    """ 
            temp, we may have write this with the dask array or from zarr image
            but for now we will use the openslide library to read the image from the wsi
    """

    def read_tiles_parallel(self,slide_path, pred_list, tile_size):
        tiles = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit the read_tile function for each tile position
            future_to_tile = {executor.submit(self.process, slide_path, pred, tile_size): pred for pred in pred_list}
            for future in concurrent.futures.as_completed(future_to_tile):
                # tile = future.result()
                tiles.append(future.result())
            print(executor._max_workers)
            executor.shutdown()
        return tiles

    def rerun_predict(self,wsi_path,pred_list,tile_size=1024):
        anote =[]
        set_logging()
        torch.cuda.empty_cache()        
        # for image in divide_chunks(images,batch_size):
        for pred_batch in utils.divide_chunks(pred_list,self.batch_size):            
            # image_batch = pool.map(process,pred_batch)        
            image_batch = self.read_tiles_parallel(wsi_path, pred_batch, tile_size)
            lefts, tops, shapes, img = self.fetch_batch(image_batch)        
            img = torch.from_numpy(img).to(self.device)                  
            img = img.half() if self.half else img.float()  # uint8 to fp16/32        
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)                
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak                        
                pred = self.model(img, augment=False)[0]        
            # Apply NMS            
            pred = non_max_suppression(pred,self.conf_thres,self.iou_thres, self.classes,False)              
            for i, det in enumerate(pred):  # detections per image                            
                left, top = lefts[i],tops[i]                
                gn = torch.tensor(shapes[i])[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    #det[:, :4] = scale_coords((img.shape[2:], det[:, :4],(shapes[i][1],shapes[i][2])).round()
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4],shapes[i]).round()                   
                    for *xyxy, conf, cls in reversed(det): 					                    
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh                        
                        line = ([*xywh])  #label format                                        
                        #print(line)
                        x1 = left + (float(line[0]) - float(line[2])/2)*(shapes[i][0])
                        y1 = top + (float(line[1]) - float(line[3])/2)*(shapes[i][1])
                        x2 = x1 + float(line[2])*shapes[i][0]
                        y2 = y1 + float(line[3])*shapes[i][1]
                        anote.append([x1,y1,x2,y2,round(float(conf),3),"path"])                        
                gn =None
                det =None
            pred = None
            del(image_batch)  
        return anote    

            



