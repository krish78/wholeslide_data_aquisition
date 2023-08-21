import openslide
from openslide.deepzoom import DeepZoomGenerator
import threading
import time
import numpy as np
from collections import deque
import gc
import copy

class data_acquisition_simulator(threading.Thread):
    def __init__(self,slide_path,que, tile_size=1024,overlap=128, total_cols=10, total_rows=10):
         super().__init__()
         self.slide_path = slide_path
         self.tile_size = tile_size
         self.Total_cols = total_cols
         self.Total_Rows = total_rows
         self.overlap = overlap
         print('simulator initialised')
         self.queue = que         
        # self.process_collector = None
        
    def run(self):
        self.read_tiles()
        

    def read_tiles(self):
        lock = threading.Lock()
        slide = openslide.open_slide(self.slide_path)
        tiles = DeepZoomGenerator(slide,tile_size=self.tile_size,overlap=self.overlap)
        cols, rows =self.Total_cols,self.Total_Rows                
        for row in range(rows):                         
            for col in range(cols):               
                tile = tiles.get_tile(tiles.level_count -1, (col,row))
                tile = tile.convert("RGB")
                tile_array = np.array(tile)#.transpose(1,0,2)
                # lock.acquire()
                self.add_item(tile_array)
                # lock.release()
            threading.Event().wait(0.02)
            
            
            # time.sleep(0.5)
    def add_item(self, item):
        self.queue.append(item)
      

