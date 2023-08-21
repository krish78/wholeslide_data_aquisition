"""
utils used in various inferences
"""
import xml.etree.ElementTree as ET
import openslide 


class utils():
    def __init__(self):
        pass 

    def replace_bigger_box(self,boxA,boxB):            
            x1 = min(boxA[0],boxB[0])
            y1 = min(boxA[1],boxB[1])
            x2 = max(boxA[2],boxB[2])
            y2 = max(boxA[3],boxB[3])
            conf = (float(boxA[4])+ float(boxB[4]))/2        
            return x1,y1,x2,y2,conf,boxA[5]

    def prune_list(self,box_list):        
            # box_list = get_box_list(wsi_path, predict_xml_path,221)
            print('before prune:',len(box_list))
            bool_list =[True for i in range(len(box_list))]    
            
            for i in range(len(box_list)-1):
                # if i == 16:
                # print('check',bool_list[i],box_list[i])
                if bool_list[i] == True:            
                    for j in range(i+1, len(box_list)):                
                        iou = self.bb_intersection_over_union(box_list[i],box_list[j])                                     
                        if iou > 0.01:                       
                            # replace rectlist{j] with largest of rectlist{i] and rectlist{j]
                            x1,y1,x2,y2,conf,path = self.replace_bigger_box(box_list[i],box_list[j])
                            box_list[j] = (x1,y1,x2,y2,conf,path) 
                            bool_list[i] = False
                            # loop continue = true
                            break
            # collect all with true
            annote_final =[]
            for i in range(len(box_list)):        
                # print('check change',i,bool_list[i],box_list[i])
                if bool_list[i] == True:
                    annote_final.append(box_list[i])
            print('after prune :',len(annote_final))
            return annote_final

    def divide_chunks(self,l, n):      
        # looping till length l
        for i in range(0, len(l), n): 
            yield l[i:i + n]

    def get_referance(self,wsi_path,nm_p=221):
        slide = openslide.open_slide(wsi_path)    
        
        w = int(slide.properties.get('openslide.level[0].width'))
        h = int(slide.properties.get('openslide.level[0].height'))
            
        ImageCenter_X = (w/2)*nm_p
        ImageCenter_Y = (h/2)*nm_p
        
        OffSet_From_Image_Center_X = slide.properties.get('hamamatsu.XOffsetFromSlideCentre')
        OffSet_From_Image_Center_Y = slide.properties.get('hamamatsu.YOffsetFromSlideCentre')
        
        # print("offset from Img center units?", OffSet_From_Image_Center_X,OffSet_From_Image_Center_Y)
        
        X_Ref = float(ImageCenter_X) - float(OffSet_From_Image_Center_X)
        Y_Ref = float(ImageCenter_Y) - float(OffSet_From_Image_Center_Y)
        slide.close()    
        #print(ImageCenter_X,ImageCenter_Y)    
        #print(X_Reference,Y_Reference)
        return X_Ref,Y_Ref

    """
    finds the interaction over union of two boxes a typical measure in object detection

    """

    def bb_intersection_over_union(self,boxA, boxB):
            # determine the (x, y)-coordinates of the intersection rectangle
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            # compute the area of intersection rectangle
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
            boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the interesection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
            # return the intersection over union value
            return iou

    def dump_results(self,gt_xml_path,predicts_xml_path,iuo_thres=0.3,nmp=221):    
        gt_box_list = self.get_box_list(gt_xml_path,nm_p=nmp)
        predict_box_list = self.get_box_list(predicts_xml_path,nm_p=nmp)    
        tp_count=0               
            
        for gt_box in gt_box_list:            
            for p_box in predict_box_list:
                iou = self.bb_intersection_over_union(gt_box,p_box)
                if iou >iuo_thres:
                    tp_count +=1
                #print(,iou)
                #total_predicts = len(predict_box_list)        
        
        fp_count =(len(predict_box_list)-tp_count)
        tp_rate = tp_count/len(gt_box_list)
        fp_rate = fp_count/len(predict_box_list)
        
        print('tp_count',tp_count,'recall or tp rate :', tp_rate)                
        print('fp_count',fp_count,'precsion or fp rate',fp_rate)                
            
    def get_box_list(xml_path,nm_p=221):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        box_list = []

        for pointlist in root.findall('pointlist'):
            x_values = [int(point.find('x').text) for point in pointlist.findall('point')]
            y_values = [int(point.find('y').text) for point in pointlist.findall('point')]

            x1, x2 = min(x_values)//nm_p, max(x_values)//nm_p
            y1, y2 = min(y_values)//nm_p, max(y_values)//nm_p

            box_list.append((x1, y1, x2, y2))

        return box_list


    def write_xml(annotations,start_id,anote_list,X_Reference,Y_Reference)     :    
        id_ = start_id
        for line in anote_list : 
            write_annotation(annotations,id_,line[0],line[1],line[2],line[3],line[4],X_Reference,Y_Reference)   
            id_ +=1
        b_xml = ET.tostring(annotations)       
        return b_xml


    def write_annotation(annotations,_id,x1,y1,x2,y2,conf,X_Reference,Y_Reference,nm_p=221):
        sub_elem  = ET.SubElement(annotations,'ndpviewstate')
        sub_elem.set('id',str(_id))
        sub_elem1 = ET.SubElement(sub_elem,'title')
        sub_elem1.text = "predict" + str(_id) # str(conf)
        sub_elem2 = ET.SubElement(sub_elem,'coordformat')    
        sub_elem2.text = 'nanometers'
        sub_elem3 = ET.SubElement(sub_elem,'lens')
        sub_elem3.text = '40.0' 
        sub_elem4 = ET.SubElement(sub_elem,'fp-tp')
        sub_elem4.text = str('none')    
        sub_elemX,sub_elemY, sub_elemZ = ET.SubElement(sub_elem,'x'), ET.SubElement(sub_elem,'y'),ET.SubElement(sub_elem,'z')
        sub_show = ET.SubElement(sub_elem,'showtitle')
        sub_show.text = str(1)
        sub_show = ET.SubElement(sub_elem,'conf')
        sub_show.text = str(conf)    

        sub_show = ET.SubElement(sub_elem,'showhistogram')
        sub_show.text = str(0)
        sub_show = ET.SubElement(sub_elem,'showlineprofile')
        sub_show.text = str(0) 
        sub_elemX.text,sub_elemY.text,sub_elemZ.text = str(int((x1+x2)*nm_p/2 -X_Reference)), str(int((y1+y2)*nm_p/2 -Y_Reference)),  '0' 
        #print(sub_elemX.text,sub_elemY.text,sub_elemZ.text)
        
        anote = ET.SubElement(sub_elem,'annotation')
        anote.set('type',"freehand")
        anote.set('displayname',"AnnotateRectangle")
        color = '#90EE90'
        if conf >= 0.5 and conf < 0.7 : 
            color    = "#9acd32"        
        elif conf >= 0.7 and conf < 0.9 :
            color = '#FFA500'
        elif conf >=0.9: 
            color = '#FFFF00'     
        anote.set('color', color)
        measure_type =ET.SubElement(anote,'measuretype')
        measure_type.text = str(3)
        Pointlist = ET.SubElement(anote, 'pointlist')
        point1 = ET.SubElement(Pointlist,'point')
        ndpa_x1 = ET.SubElement(point1,'x')
        ndpa_y1 = ET.SubElement(point1,'y')
        
        ndpa_x1.text = str(int(x1*nm_p-X_Reference)) 
        ndpa_y1.text = str(int(y1*nm_p-Y_Reference))
        
        
        #print(ndpa_x1.text,ndpa_y1.text)    
        point2 = ET.SubElement(Pointlist,'point')
        
        ndpa_x2 = ET.SubElement(point2,'x')
        ndpa_y2 = ET.SubElement(point2,'y')     
        
        ndpa_x2.text = ndpa_x1.text
        ndpa_y2.text = str(int(y2*nm_p-Y_Reference))
        
        point3 = ET.SubElement(Pointlist,'point')
        ndpa_x3 = ET.SubElement(point3,'x')
        ndpa_y3 = ET.SubElement(point3,'y')
        ndpa_x3.text = str(int(x2*nm_p -X_Reference))
        
        ndpa_y3.text = ndpa_y2.text
            
        point4 = ET.SubElement(Pointlist,'point')
        
        ndpa_x4 = ET.SubElement(point4,'x')
        ndpa_y4 = ET.SubElement(point4,'y')                            
        ndpa_x4.text = ndpa_x3.text
        ndpa_y4.text = ndpa_y1.text
            
        anote_type =ET.SubElement(anote,'specialtype')
        anote_type.text = 'rectangle'
        anote_type =ET.SubElement(anote,'closed')
        anote_type.text = '1'     
# end of write_annotation function



