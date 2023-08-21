import dask.array as da
import threading
import numpy as np
import xml.etree.ElementTree as ET
from tifffile import imwrite
import os
import time
from utils.datasets import letterbox
from inference import inference
from inference_utils import utils 

class inference_io(threading.Thread):    # lock = threading.Lock()    
    def __init__(self,wsi_path,dump_path,wsi_dask,total_cols,total_rows,frame_size,tile_size=512,overlap=64,batch_size=8):    
        super().__init__()
        self.wsi_path = wsi_path
        self.xml_path = os.path.join(dump_path,'annotation.xml')
        self.batch_size = batch_size       
        self.dump_path = dump_path 
        self.wsi_dask = wsi_dask
        
        self.tile_size = tile_size
        self.overlap = overlap#int(overlap_pc*tile_size/100)
        
        self.total_frames = total_cols * total_rows
        self.total_cols = total_cols* frame_size//tile_size
        self.total_rows = total_rows* frame_size//tile_size        
        
        self.total_tiles = self.total_cols * self.total_rows
        
        self.tiles_processed = 0
        self.batch_number = 0
        self.batch =[]        
        self.inference = inference(dump_path,batch_size=batch_size)

    def run(self):
        self.process_wsi()        
    
    def is_tile_ready(self,row,col):                     
        x1, y1, x2,y2 = self.get_tile_position(row,col)
        
        # if outer_x1 <= inner_x1 <= inner_x2 <= outer_x2 and outer_y1 <= inner_y1 <= inner_y2 <= outer_y2:
        # can check on the zero side.
        
        # if self.batch_number > 150:
        #     print('dask_check', x2,self.wsi_dask.processed_da.shape[0], y2,self.wsi_dask.processed_da.shape[1])
        if x2<=self.wsi_dask.processed_da.shape[0] and y2<=self.wsi_dask.processed_da.shape[1]:
            return True                    
        else:
            return False    
    # prepare postions for getting the tile with overlap
    def get_tile_position(self,row,col):        
        x_start, y_start = row*self.tile_size, col*self.tile_size
        x_end, y_end = (row+1)*self.tile_size, (col+1)*self.tile_size    
        if x_start > self.overlap:
            x_start = x_start -self.overlap
        if y_start > self.overlap:
            y_start = y_start -self.overlap   
        if x_end + self.overlap <= self.total_rows*self.tile_size:
            x_end = x_end + self.overlap
        if y_end + self.overlap <= self.total_cols*self.tile_size:
            y_end = y_end + self.overlap       

        return x_start,y_start,x_end,y_end    

    def get_tile(self,row,col):
        # row = self.tiles_processed//self.total_cols                
        # col = self.tiles_processed%self.total_cols
        # if self.batch_number > 35:
        #     print('col,row',col,row)
        x_start,y_start,x_end,y_end = self.get_tile_position(row,col)
        tile = self.wsi_dask.processed_da[x_start:x_end,y_start:y_end,:]        
        # imwrite(os.path.join(self.dump_path,'tile_{0}_{1}.tif'.format(row,col)),np.array(tile))
        shape = (tile.shape[0],tile.shape[1])
        img = letterbox(np.array(tile),auto=False)[0]
        img = img.transpose(2,0,1)
        img = np.expand_dims(img, 0) 
        # may have to convert to rgb
        return y_start,x_start,shape,img

    def get_batch_over(self):    
        batch_iter = 0    
        batch_done = False
        self.batch =[]
        # print('batch_size',self.batch_size)
        while batch_iter < self.batch_size:
            row = self.tiles_processed//self.total_cols                
            col = self.tiles_processed%self.total_cols
            if self.is_tile_ready(row,col):                
                self.batch.append(self.get_tile(row,col))
                self.tiles_processed +=1
                batch_iter+=1
            else:
                threading.Event().wait(0.05)            
        # threading.Event().wait(0.05)
        self.batch_number +=1        
        # self.frames_got += self.batch_size
        batch_done = True
        # print('tiles processed',self.tiles_processed)
        return batch_done

    def process_wsi(self):
        anote =[]
        print(self.total_tiles)
        # while(self.tiles_processed < self.total_tiles) or \
        self.batch_number =0
        print('total batches =', self.total_tiles//self.batch_size)
        try:
            while (self.batch_number < self.total_tiles//self.batch_size):                                            
                if self.get_batch_over():                             
                    anote = self.inference.run_batch_inference(self.batch,anote,self.wsi_path)
                    print('whiling here at batch :',self.batch_number)
                else:
                    threading.Event().wait(0.1)            
                threading.Event().wait(0.005)        
        except Exception as e:
            print('batches done before crash',self.batch_number)
            print('error in inference thread', e)            

        print('out of inference thread')
        print('batches done',self.batch_number)
        anote_final = utils.prune_list(anote)            
        self.write_inference(anote_final)
        # anote_rerun =  self.inference.rerun_predict(self.wsi_path,anote_final)            
        # anote_updated = prune_list(anote_rerun)                   
        # self.write_inference(anote_updated)

    def write_inference(self,anotes):
        annotations = ET.Element('annotations')
        start_id = 0
        x_ref, y_ref = utils.get_referance(self.wsi_path)
        anote_xml = utils.write_xml(annotations,start_id,anotes,x_ref,y_ref)                               
        with open(self.xml_path, "wb") as f:
                  f.write(anote_xml)

    
           
# inference from model
