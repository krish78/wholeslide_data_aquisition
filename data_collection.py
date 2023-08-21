from collections import deque
import cv2
import numpy as np
import dask.array as da
import zarr
import threading
import copy
import time


from tifffile import imwrite
# from data_acqusition_simulator import data_acquisition_simulator
"""

"""
# num_processed_frames = 0

class ProcessCollection(threading.Thread):
    # register = register() # this class to be created and initialised    
    
    def __init__(self,queue,wsi_dask,tile_size=1024, total_cols=10, total_rows=10):
        super().__init__()        
        self.wsi_dask = wsi_dask
        self.tile_size = tile_size        
        self.Total_cols = total_cols
        self.Total_Rows = total_rows
        self.Total_frames = total_cols *total_rows
        self.Min_Required_frames = total_cols + 1                    
        self.queue = queue                
        

    def add_item(self, item):
        # self.queue.appendleft(item)
        self.queue.append(item)

    def remove_item(self):
        if self.queue:
            return self.queue.popleft()
        else:
            raise IndexError("The collection is empty.")

    def peek(self, index):
        if self.queue:
            return self.queue[index]
        else:
            raise IndexError("The collection is empty.")

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)

    # If this returns true caller should call clear data
    # def IsProcess_complete(self):
    #         # return self.Num_Processed_Frames >= self.Total_frames        
    #     return num_processed_frames >= self.Total_frames
    
    def trim(self, lefttrim, topttrim, width, height, image):
        # Assuming the input image is a 3D NumPy array with shape (height, width, 3)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image must be a 2D RGB image")

        # Get the dimensions of the input image
        image_height, image_width, _ = image.shape # overlapped image so ex 1280,1280
        
        righttrim = lefttrim + width
        bottomtrim = topttrim + height

        # Ensure the trimming parameters are within valid bounds
        lefttrim = int(max(0, lefttrim))
        righttrim = int(min(image_width, righttrim))
        toptrim = int(max(0, topttrim))
        bottomtrim = int(min(image_height, bottomtrim))
        # print(lefttrim, righttrim, topttrim, bottomtrim)

        # Trim the image based on the specified parameters
        # trimmed_image = copy.deepcopy(image[toptrim:bottomtrim, lefttrim:righttrim,:])       
        trimmed_image = image[toptrim:bottomtrim, lefttrim:righttrim,:]       
        return trimmed_image

    def Find_Offsets(self,target_image, ref_image ):
        # Assuming the input images are 2D RGB images represented as NumPy arrays
        if len(ref_image.shape) != 3 or ref_image.shape[2] != 3 or \
           len(target_image.shape) != 3 or target_image.shape[2] != 3:
            raise ValueError("Input images must be 2D RGB images")
        ref_gray = cv2.cvtColor(ref_image, cv2.COLOR_RGB2GRAY)
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
        result = cv2.phaseCorrelate(np.float32(ref_gray), np.float32(target_gray))
        offset_x, offset_y = int(abs(result[0][1]/2)) , int(abs(result[0][0]/2)) # x and y are swapped in phaseCorrelate
        print(offset_x,offset_y)
        return offset_x,offset_y

    def run(self):
        self.process_frames_thread()        
        # threading.Event().wait(0.3)
        
    def process_frames_thread(self):        
        #exit incase of user action or exceptions 
        num_processed_frames = int(self.wsi_dask.num_processed_frames)
        
        while (num_processed_frames < self.Total_frames) :        
        # while (self.Num_Processed_Frames < self.Total_frames) :        
            length = len(self.queue)

            if (length >= self.Total_cols * 2):                 

                self.register_raster_group()                       
                for _ in range(self.Total_cols):
                    item = self.remove_item()     

            else:
                # Sleep briefly if there are not enough frames to process                
                threading.Event().wait(0.05)    
                # time.sleep(0.5)
            # time.sleep(0.01)
            threading.Event().wait(0.02) 
            num_processed_frames = int(self.wsi_dask.num_processed_frames)           
        print('out of loop process frames thread')                
        # self.stop_processing()
        return True    
    def process_first_col(self):
        # row_count = self.Num_Processed_Frames // self.Total_cols
        row_count = int(self.wsi_dask.num_processed_frames) // self.Total_cols
        if (row_count == 0):
            offset_x = 0
            offset_y = 0 
            start = 0
        else:
            # try:
            #     offset_x, offset_y = self.Find_Offsets(self.peek(0), self.peek(self.Total_cols))
            # except:
            offset_x = 0 #this should come from registration
            offset_y = 128 #this should come from registration
            start = self.Total_cols
        # print(offset_x,offset_y)        
        trimmed_image = self.trim(offset_x, offset_y,self.tile_size,self.tile_size, self.peek(start))
        # col_index = 0
        # row_index = row_count*self.tile_size            
        # self.processed_images[row_index:row_index+self.tile_size,col_index:col_index+self.tile_size,:] = trimmed_image   
        # self.processed_images[row_index:row_index+self.tile_size,col_index:col_index+self.tile_size,:] = copy.deepcopy(trimmed_image)     
        # da_row = None
        da_row = da.from_array(trimmed_image,chunks=(self.tile_size,self.tile_size,3))
        # del trimmed_image
        # self.Num_Processed_Frames += 1
        return offset_x, offset_y, da_row
        

    def process_row(self):                       
        # row_count = self.Num_Processed_Frames // self.Total_cols
        
        row_count = int(self.wsi_dask.num_processed_frames) // self.Total_cols
        # group_horizantal_shift, group_vertical_shift  = self.process_first_col()
        group_horizantal_shift, group_vertical_shift, da_row = self.process_first_col()
        # print('first col of', row_count, group_horizantal_shift, group_vertical_shift)
        if (row_count == 0):
            start = 0
        else:
            start = self.Total_cols

        for i in range(start+1, start + self.Total_cols):
            try:
                off_x, off_y = self.Find_Offsets(self.peek(i - 1), self.peek(i))
            except:            # print('offsets horizontal',off_x, off_y)
                off_x, off_y = 128 , 0
            offset_x = int(off_x) + group_horizantal_shift
            offset_y = int(off_y) + group_vertical_shift                                   
            trimmed_image = self.trim(offset_x, offset_y,self.tile_size,self.tile_size,self.peek(i))
            # print(trimmed_image.shape)
            # imwrite('E:\cytology\dump\image_{0}{1}.tif'.format(row_count,i),trimmed_image)
            # col_index = (i-start)*self.tile_size
            # row_index = row_count*self.tile_size      
            # self.processed_images[row_index:row_index+self.tile_size,col_index:col_index+self.tile_size,:] = trimmed_image       
            # self.processed_images[row_index:row_index+self.tile_size,col_index:col_index+self.tile_size,:] = trimmed_image.copy() 
            da_row = da.hstack([da_row,da.from_array(trimmed_image,chunks=(self.tile_size,self.tile_size,3))])            
            # da_row = da.append(trimmed_image,da_row,axis=1)
            # np_row = np.hstack([np_row,trimmed_image])
            # del trimmed_image      
            # print(da_row.shape)
       
        # self.Num_Processed_Frames += self.Total_cols
        
        self.wsi_dask.num_processed_frames += self.Total_cols
        # da_row.to_zarr('E:\cytology\dump\image_{0}.zarr'.format(row_count))
        
        # if (self.Num_Processed_Frames <= self.Total_cols):   
                
        if (int(self.wsi_dask.num_processed_frames) <= self.Total_cols):
            # self.processed_images = da.vstack((self.processed_images,da_row))
            self.wsi_dask.processed_da = da_row
        else:            
            # self.processed_images = da.vstack((self.processed_images,da_row))
            self.wsi_dask.processed_da = da.vstack((self.wsi_dask.processed_da,da_row))
            
        # print(self.processed_images.shape) 
        # self.processed_images.compute()
        # print(self.Num_Processed_Frames, 'frames processed')
        # print(self.wsi_dask.num_processed_frames, 'frames processed')


    def register_raster_group(self):
        # Assuming this method performs registration for a group of frames horizontally        
        # if (self.Num_Processed_Frames == 0):
        if (int(self.wsi_dask.num_processed_frames) == 0):
            self.process_row()
        self.process_row()    
        return True
    
    # def save_processed_images(self,path):  
          
    #     # self.processed_images.compute()
    #     import os
    #     import shutil
    #     if os.path.isdir(path):
    #         shutil.rmtree(path)
    #     # print(self.processed_images.shape)
    #     # self.processed_images.to_zarr(path)
    #     self.wsi_dask.processed_da.to_zarr(path)