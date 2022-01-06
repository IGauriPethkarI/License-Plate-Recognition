import cv2
import imutils
import numpy as np
import pytesseract as tess
import tkinter as tk
import PIL.Image, PIL.ImageTk
from tkinter import filedialog

tess.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#State Dictionary
States = {"AN" : "Andaman and Nicobar", "AP" : "Andhra Pradesh", "AR" : "Arunachal Pradesh",
          "AS" : "Assam","BR" : "Bihar","CH" : "Chandigarh","DN" : "Dadra and Nagar Haveli",
          "DD" : "Daman and Diu","DL" : "Delhi","GA" : "Goa","GJ" : "Gujarat","HR" : "Haryana",
          "HP" : "Himachal Pradesh","JK" : "Jammu and Kashmir","KA" : "Karnataka","KL" : "Kerala",
          "LD" : "Lakshadweep","MP" : "Madhya Pradesh","MH" : "Maharashtra","MN" : "Manipur",
          "ML" : "Meghalaya","MZ" : "Mizoram","NL" : "Nagaland","OR" : "Orissa","PY" : "Pondicherry",
          "PN" : "Punjab","RJ" : "Rajasthan","SK" : "Sikkim","TN" : "TamilNadu","TR" : "Tripura",
          "UP" : "Uttar Pradesh","WB" : "West Bengal",
}

#Stacking images for the output
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def plateRecognition(imgOrignal):    
    #Original Image
    img = cv2.imread(imgOrignal)

    #Gray Scale image
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Reduce noise and make the image smooth
    imgSmooth = cv2.bilateralFilter(imgGray, 11, 17, 17)

    #Find Edges
    imgEdged = cv2.Canny(imgSmooth, 170, 200)

    #Threshold Image
    kernel = np.ones((2, 2), np.uint8)
    imgDilate = cv2.dilate(imgEdged, kernel, iterations=1)
    imgThreshold = cv2.erode(imgDilate, kernel, iterations=1)

    #Find Contours
    contours, hierarchy = cv2.findContours(imgThreshold.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    imgContour = img.copy()
    cv2.drawContours(imgContour, contours, -1, (255,51,25), 2)  

    #Minimizing the Contours
    contours = sorted(contours, key= cv2.contourArea, reverse= True)[:30]
    NumberPlate = None
    imgDefined = img.copy()
    cv2.drawContours(imgDefined, contours, -1, (255,51,25), 2)

    #Finding the number plate
    count = 0
    tag = 1

    for i in contours:
        perimeter = cv2.arcLength(i,True)
        approx = cv2.approxPolyDP(i, 0.02*perimeter, True)

        if (len(approx) == 4):
            NumberPlate = approx
            #cropping the image
            a, b = (int(0.02*img.shape[0]), int(0.025*img.shape[1]))
            x, y, w, h = cv2.boundingRect(i)
            imgCrop = img[y+a:y+h-a, x+b:x+w-b]

            cv2.imwrite(str(tag)+ '.png', imgCrop)
            tag += 1
            break
        
    #Draw Contours for Number Plate
    imgBoundRect = img.copy()
    cv2.drawContours(imgBoundRect, [NumberPlate], -1, (255, 25, 52), 2)

    #Cropped Number Plate
    imgPlateCrop = "1.png"
    cv2.imshow("Cropped Plate Image", cv2.imread(imgPlateCrop))
    cv2.waitKey(0)

    #Find text on the Number Plate
    global Text
    Text = tess.image_to_string(imgPlateCrop, lang = 'eng')
    print("The Number Plate is given by: ", Text)

    #Find the State that the vehicle belongs to
    Text = ''.join(e for e in Text if e.isalnum())
    stat = Text[0:2]

    try:
        print("Car belongs to ", States[stat])
    except:
        print("State not recognised")


    #Final Image
    imgResult = img.copy()
    cv2.rectangle(imgResult, (x,y), (x+w, y+h), (213,183,163), 2)
    cv2.rectangle(imgResult, (x,y-40), (x+w, y), (213,183,163), -10)
    cv2.putText(imgResult, Text, (x,y-10), cv2.FONT_ITALIC, 0.7, (0,0,0),2)
    cv2.imwrite("Result.jpg", imgResult)

    #Flow of the Plate Detection
    cv2.putText(img, "Original Image", (20,20), cv2.FONT_ITALIC, 0.8, (0,0,0),2)
    cv2.putText(imgSmooth, "GrayScale Smooth Image", (20,20), cv2.FONT_ITALIC, 0.5, (0,0,0), 2)
    cv2.putText(imgEdged, "Canny Image", (20,20), cv2.FONT_ITALIC, 0.8, (255,255,255), 2)
    cv2.putText(imgThreshold, "Threshold Image", (20,20), cv2.FONT_ITALIC, 0.8, (255,255,255), 2)
    cv2.putText(imgContour, "Contours Image", (20,20), cv2.FONT_ITALIC, 0.8, (0,0,0), 2)
    cv2.putText(imgDefined, "Defined Contours Image", (20,20), cv2.FONT_ITALIC, 0.5, (0,0,0), 2)
    cv2.putText(imgBoundRect, "Plate Detection Image", (20,20), cv2.FONT_ITALIC, 0.5, (0,0,0), 2)
    cv2.putText(imgResult, "Final Image", (20,20), cv2.FONT_ITALIC, 0.8, (0,0,0), 2)

    imgStack = stackImages(0.9,[[img, imgSmooth, imgEdged, imgThreshold], [imgContour, imgDefined, imgBoundRect, imgResult ]])
    cv2.imshow("Final Image", imgStack)
    cv2.imwrite("FlowImage.jpg", imgStack)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#GUI 
window = tk.Tk()
window.title("License Plate Detection")
window.geometry('900x700')
window.resizable(width = False, height = False)
window.configure(background = "#839dc6")

title = tk.Label(text = "License Plate Detection System")
title.configure(background='#3f5d8a', foreground='white', font = ('times', 30, 'bold'), width = 900, height = 2)
title.pack()

canvas1 = tk.Canvas(window, width = 340, height = 270)
canvas1.pack()
canvas1.configure(background = "#3f5d8a")
canvas1.place(x = 70, y = 110)

canvas2 = tk.Canvas(window, width = 340, height = 270)
canvas2.pack()
canvas2.configure(background = "#3f5d8a")
canvas2.place(x = 450, y = 110)

canvas3 = tk.Canvas(window, width = 700, height = 220)
canvas3.pack()
canvas3.configure(background = "#3f5d8a")
canvas3.place(x = 70, y = 460)

l1 = tk.Label(window, text = "The Number Plate:",font = ('times', 12, 'bold'), fg = 'white', bg = "#3f5d8a") 
l1.place(x = 490, y = 112)

def result():
    
    label1 = tk.Label(window, text = Text, font = ('times', 17, 'bold'), fg = 'white', bg = "#3f5d8a")
    label1.place(x = 510, y = 130)
    
    img = PIL.Image.open("Result.jpg")
    img = img.resize((278,200), PIL.Image.ANTIALIAS)
    img = PIL.ImageTk.PhotoImage(img)
    canvas = tk.Label(window, image = img)
    canvas.image = img
    canvas.pack()
    canvas.place(x = 478, y = 167)
    
def uploadImage():
    x = filedialog.askopenfilename()
    img = PIL.Image.open(x)
    img = img.resize((278,230), PIL.Image.ANTIALIAS)
    img = PIL.ImageTk.PhotoImage(img)
    canvas = tk.Label(window, image = img)
    canvas.image = img
    canvas.pack()
    canvas.place(x = 102, y = 130)

    plateRecognition(x)

    
def flowImage():
    img = PIL.Image.open("FlowImage.jpg")
    img = img.resize((650,200), PIL.Image.ANTIALIAS)
    img = PIL.ImageTk.PhotoImage(img)
    canvas = tk.Label(window, image = img)
    canvas.image = img
    canvas.pack()
    canvas.place(x = 95, y = 470)
    

upload = tk.Button(window, text="Upload Image", command=uploadImage, padx=10, pady=5)
upload.configure(background='#3f5d8a', foreground='white',font=('times',15,'bold'))
upload.pack()
upload.place(x = 120, y = 400)

change = tk.Button(window, text="Configure",command = result, padx = 20, pady=5)
change.configure(background='#3f5d8a', foreground = 'white',font=('times',15,'bold'))  
change.pack()
change.place(x = 320, y = 400)

flow = tk.Button(window, text="Flow of the configuration",command = flowImage, padx = 5, pady=5)
flow.configure(background='#3f5d8a', foreground = 'white',font=('times',15,'bold'))  
flow.pack()
flow.place(x = 500, y = 400)

window.mainloop()
