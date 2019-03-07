import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.collections import PatchCollection
import cv2
import pdb

class RandomBoxMap:
    def __init__(self, rows=224, cols=224, dpi=112, nboxes=(0,5), n_heading = 4, slant_scale = 1, thick=40, thick_scale=2):
        img = np.zeros((rows,cols))

        patches=[]

        thick = np.random.normal(scale=thick_scale)+thick
        xy = (np.random.normal(loc=-thick/2, scale=1),np.random.normal(loc=-thick/2, scale=1)) #randint(-20,-10), np.random.randint(-20,-10))
        wh = (cols*1.2, thick)
        angle = np.random.normal(scale=slant_scale)
        wall = Rectangle(xy, wh[0],wh[1], angle=angle)
        patches.append(wall)

        xy = (np.random.normal(loc=-thick/2, scale=1),np.random.normal(loc=rows-thick/2, scale=1))
              #np.random.randint(-20,-10), np.random.randint(rows-20, rows-10))
        wh = (cols*1.2, thick)
        angle = np.random.normal(scale=slant_scale)
        wall = Rectangle(xy, wh[0],wh[1], angle=angle)
        patches.append(wall)
        
        xy = (np.random.normal(loc=-thick/2, scale=1),np.random.normal(loc=-thick/2, scale=1))
        #np.random.randint(-20,-10), np.random.randint(-20, -10))
        wh = (thick, rows*1.2)
        angle = np.random.normal(scale=slant_scale)
        wall = Rectangle(xy, wh[0],wh[1], angle=angle)
        patches.append(wall)

        xy = (np.random.normal(loc=cols-thick/2, scale=1),np.random.normal(loc=-thick/2, scale=1))
        #np.random.randint(cols-20,cols-10), np.random.randint(-20, -10))
        wh = (thick, rows*1.2)
        angle = np.random.normal(scale=slant_scale)        
        wall = Rectangle(xy, wh[0],wh[1], angle=angle)
        patches.append(wall)
        
        # xy = (np.random.randint(5,35), np.random.randint(5,35))
        # room_height = np.random.randint(180,200)
        # room_width = np.random.randint(180,200)
        # angle = np.random.randint(-5,5)
        # room = Rectangle(xy, room_height, room_width, angle=angle)
        # print (xy, room_height, room_width)
        
        fig = plt.figure(figsize=(cols/dpi,rows/dpi), dpi=dpi)
        ax = fig.gca()
        ax.imshow(img,cmap=plt.cm.binary)
        # ax.add_artist(room)
        ax.add_collection(PatchCollection(patches))#,cmap=plt.cm.binary))
        ax.set_xlim([0,cols])
        ax.set_ylim([0,rows])
        # ax.axis('tight')
        plt.subplots_adjust(0,0,1,1,0,0)
        fig.canvas.draw()        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        w, h = fig.canvas.get_width_height()
        data = data.reshape((h,w,3))
        # data = cv2.resize(data, (cols,rows))
        data = np.sum(data, axis=2)
        if np.max(data)>0:
            data = np.round(data/np.max(data))
        plt.close()
        self.room = data
        
        ## generate random boxes
        patches=[]
        for i in range(np.random.randint(nboxes[0],nboxes[1])):
            rs = min(rows,cols)
            hbox = np.random.randint(rs/10, rs/2)
            wbox = np.random.randint(rs/10, rs/2)
            xy = (np.random.randint(0,cols), np.random.randint(0,rows))
            angle = np.random.normal(scale=3)+np.random.randint(n_heading)*360/n_heading
            box = Rectangle(xy, hbox,wbox,angle=angle)
            patches.append(box)
            
        fig = plt.figure(figsize=(cols/dpi,rows/dpi), dpi=dpi)
        ax = fig.gca()
        ax.imshow(img,cmap=plt.cm.binary)
        ax.add_collection(PatchCollection(patches))#,cmap=plt.cm.binary))
        ax.set_xlim([0,cols])
        ax.set_ylim([0,rows])
        # ax.axis('tight')
        plt.subplots_adjust(0,0,1,1,0,0)
        fig.canvas.draw()        
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        w, h = fig.canvas.get_width_height()

        data = data.reshape((h,w,3))
        # data = cv2.resize(data, (cols,rows))
        data = np.sum(data, axis=2)
        if np.max(data)>0:
            data = np.round(data/np.max(data))
        plt.close()

        self.boxes = data

    def get_map(self):
        return np.clip(1-self.room+1-self.boxes,0,1)

class PartitionSpace:
    def __init__(self, rows=480,cols=480, rooms_row=(2,3), rooms_col=(1,2), slant_scale=0, n_boxes=(1,6), thick=40, thick_scale=2.0):
        # N[1,2,3] x M[1,2,3] partitioned space
        N = np.random.randint(rooms_row[0], rooms_row[1])
        M = np.random.randint(rooms_col[0], rooms_col[1])
        self.rows=rows
        self.cols=cols
        self.n_rows = N
        self.n_cols = M
        self.thick = thick
        
        prows = rows/N
        pcols = cols/M
        dpi = 40

        min_n_boxes = n_boxes[0]
        max_n_boxes = n_boxes[1]

        img = np.ones((rows,cols))
        for i in range(N):
            for j in range(M):
                rbm=RandomBoxMap(rows=prows, cols=pcols,
                                 dpi=dpi,
                                 nboxes = (min_n_boxes, max_n_boxes),
                                 n_heading = np.random.randint(1,4)*4,
                                 slant_scale=slant_scale,
                                 thick=thick, thick_scale=thick_scale)
                img[i*prows:(i+1)*prows, j*pcols:(j+1)*pcols] = rbm.get_map()
        self.result = img

    def get_map(self, rows, cols):
        the_map = cv2.resize(self.result, (rows,cols), interpolation=cv2.INTER_NEAREST)
        the_map[0,:]=1
        the_map[-1,:]=1
        the_map[:,0]=1
        the_map[:,-1]=1
        
        return the_map

    def connect_rooms(self, p_open = 0.5):
            
        N = self.n_rows
        M = self.n_cols
        prows = self.rows/N
        pcols = self.cols/M
        blank_thick = int(1.2*self.thick)
        for i in range(N):
            for j in range(M):
                if i+1 < N and np.random.rand()<=p_open:
                    #open N-S
                    r0 = (i+1)*prows - blank_thick
                    r1 = (i+1)*prows + blank_thick
                    c0 = j*pcols + pcols/3 + np.random.randint(-pcols/6,pcols/6)
                    c1 = (j+1)*pcols - pcols/3  + np.random.randint(-pcols/6,pcols/6)                   
                    self.result[r0:r1, c0:c1]=0
                if j+1 < M and np.random.rand()<=p_open:
                    #open E-W
                    r0 = i*prows + prows/3  + np.random.randint(-prows/6,prows/6)
                    r1 = (i+1)*prows - prows/3 + np.random.randint(-prows/6,prows/6)
                    c0 = (j+1)*pcols - blank_thick
                    c1 = (j+1)*pcols + blank_thick
                    self.result[r0:r1, c0:c1]=0

if __name__ == '__main__':
    plt.figure()                    
    while 1:
        #rooms_row: number of rooms in a row [a,b): a <= n < b
        #rooms_col: number of rooms in a col [a,b): a <= n < b

        kwargs = {'rooms_row':(1,2), 'rooms_col':(2,3),
                  'slant_scale':2, 'n_boxes':(1,8), 'thick':50, 'thick_scale':3}
        ps = PartitionSpace(**kwargs)

        # p_open : probability to have the doors open between rooms
        ps.connect_rooms(p_open=1.0)

        # set output map size
        X=ps.get_map(88,88)

        plt.imshow(X, interpolation='nearest', cmap=plt.cm.binary)
        plt.pause(1)
