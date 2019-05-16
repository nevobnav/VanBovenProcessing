# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:59:27 2019

@author: VanBoven
"""

from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
import cv2



coords = []
#straal vanaf het midden tot uiterste rand van object
plant_straal = 15
#categorie, kies uit Broccoli, Background, Grass, Cover etc.
cat = 'Background'


if __name__ == "__main__":
    root = Tk()

    #setting up a tkinter canvas with scrollbars
    frame = Frame(root, bd=2, relief=SUNKEN)
    frame.grid_rowconfigure(0, weight=1)
    frame.grid_columnconfigure(0, weight=1)
    xscroll = Scrollbar(frame, orient=HORIZONTAL)
    xscroll.grid(row=1, column=0, sticky=E+W)
    yscroll = Scrollbar(frame)
    yscroll.grid(row=0, column=1, sticky=N+S)
    canvas = Canvas(frame, bd=0, xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
    canvas.grid(row=0, column=0, sticky=N+S+E+W)
    xscroll.config(command=canvas.xview)
    yscroll.config(command=canvas.yview)
    frame.pack(fill=BOTH,expand=1)

    #adding the image
    File = askopenfilename(parent=root, initialdir="C:/",title='Choose an image.')
    img = ImageTk.PhotoImage(Image.open(File))
    canvas.create_image(0,0,image=img,anchor="nw")
    canvas.config(scrollregion=canvas.bbox(ALL))

    #function to be called when mouse is clicked
    def printcoords(event):
        #outputting x and y coords to console
        coords.append([event.x, event.y])
        
    #mouseclick event
    canvas.bind("<Button 1>",printcoords)

    root.mainloop()
       
img = cv2.imread(str(File), -1)

for x, y in coords:
    output = img[y-plant_straal: y+plant_straal,x-plant_straal:x+plant_straal]
    cv2.imwrite(r'E:\400 Data analysis\410 Plant count\Training_data/' +str(cat) +'/click_' + str(File[-22:-4]) + str(x)+str(y)+'.jpg', output)