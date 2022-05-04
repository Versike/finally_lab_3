import math
import tkinter.messagebox
from asyncio.windows_events import NULL
from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import methods as clm
import sobel_danromkit as filters
import copy

plt.rcParams.update({'font.size': 5.5})


class Main:
    def __init__(self, root):
        self.root = root
        self.baseImg = None
        self.img = np.zeros([])
        self.oriImg = np.zeros([])
        self.saveImg = np.zeros([])

        # Image mode = 0, Histogram mode = 1
        self.viewMode = 0
        self.currentMode = NULL
        self.is_firstImg = TRUE
        self.is_grayscale = FALSE

        # Prepare Window Settings
        self.screenWidth = self.root.winfo_screenwidth()
        self.screenHeight = self.root.winfo_screenheight()
        self.root.title('КЗ | Дамир Рома')
        self.root.geometry("%dx%d" % (self.screenWidth, self.screenHeight))
        self.root.option_add('*tearOff', FALSE)

        # Set up panels to show the image
        self.imgPanelSize = [self.screenWidth // 1.2, self.screenHeight // 1.2]
        self.imgPanelSizeHist = [self.screenWidth // 2, self.screenHeight // 2]

        self.panelLeft = tk.Label(self.root, width=int(self.screenWidth // 2.15), height=int(self.screenHeight // 1.6),
                                  relief="ridge")
        self.panelLeft.grid(row=1, column=0, ipadx=20, ipady=20)

        self.panelRight = tk.Label(self.root, width=int(self.screenWidth // 2.15), height=int(self.screenHeight // 1.6),
                                   relief="ridge")
        self.panelRight.grid(row=1, column=1, ipadx=20, ipady=20)

        # Initialize View Mode Information
        self.currentMode = tk.Label(self.root, text="Лабораторная 3")
        self.currentMode.config(font=("Courier", 14))
        self.currentMode.grid(row=0, column=0, columnspan=2)

        # Initialize Histogram Figures
        self.histFigure = plt.Figure(figsize=(5.2, 5.2), dpi=100)
        self.histCanvas = FigureCanvasTkAgg(self.histFigure, self.root)

        # Set up needed action menus
        self.menubar = Menu(self.root)

        ############################# FILE MENUS #############################
        self.fileMenu = Menu(self.menubar)
        self.menubar.add_cascade(label="Файл", menu=self.fileMenu)
        self.fileMenu.add_command(label="Открыть", command=self.selectImage)
        self.fileMenu.add_command(label="Сохранить", command=self.saveImage)
        self.fileMenu.add_separator()
        self.fileMenu.add_command(label="Закрыть", command=self.root.quit)

        ############################# VIEW MODE MENUS #############################
        self.viewMenu = Menu(self.menubar)
        self.menubar.add_cascade(label="Вид анализа", menu=self.viewMenu)
        self.viewMenu.add_command(label="Сравнение", command=self.toImgMode)
        self.viewMenu.add_command(label="Гистограмма", command=self.toHistMode)

        self.showBottomPanel()
        self.root.config(menu=self.menubar)
        self.root.mainloop()

    # Handle image selection
    def selectImage(self):
        filename = filedialog.askopenfilename(filetypes=[("Image files", ".jpg .jpeg .jp2 .png .tiff .svg .gif .bmp")])
        if not filename:
            return
        else:
            self.clearLeftPanel()
            self.clearRightPanel()
            self.clearHistogramCanvas()
            self.img = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
            self.oriImg = cv.cvtColor(cv.imread(filename), cv.COLOR_BGR2RGB)
            image = Image.fromarray(self.img)
            self.baseImg = Image.open(filename)

            self.setSize(image)
            image_tk = ImageTk.PhotoImage(image)

            if (self.viewMode == 0):
                self.showLeftPanel(image_tk)
                self.showRightPanel(image_tk)
            elif (self.viewMode == 1):
                self.clearLeftPanel()
                self.showLeftPanel(image_tk)
                self.showHistogramCanvas()
            self.is_firstImg = FALSE

    def saveImage(self):
        filename = filedialog.asksaveasfile(mode='w', defaultextension=".jpg",
                                            filetypes=[("Image files", ".jpg .jpeg .jp2 .png .tiff .svg .gif .bmp")])
        if not filename:
            return
        self.saveImg.save(filename)

    def setImgArray(self, img):
        self.saveImg = img
        self.img = np.array(img)

    # Reset effect
    def clearEffect(self):
        self.handleGray(self.img)
        self.img = self.oriImg
        img = Image.fromarray(self.img)
        self.setSize(img)
        img = ImageTk.PhotoImage(img)
        self.clearLeftPanel()
        self.clearRightPanel()
        self.showLeftPanel(img)
        self.clearHistogramCanvas()
        if (self.viewMode == 0):
            self.showRightPanel(img)
        else:
            self.showHistogramCanvas()

    # Handle Display Image
    def showImage(self, img):
        if (self.viewMode == 0):
            oriImg = Image.fromarray(self.oriImg)
            self.setSize(oriImg)
            oriImg = ImageTk.PhotoImage(oriImg)
            self.showLeftPanel(oriImg)
            self.showRightPanel(img)
        elif (self.viewMode == 1):
            self.clearLeftPanel()
            self.showLeftPanel(img)
            self.showHistogramCanvas()

    # Handle thumbnail size
    def setSize(self, img):
        img.thumbnail(self.imgPanelSize, Image.ANTIALIAS)

    def handleGray(self, img):
        if (self.is_grayscale == TRUE):
            self.img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            self.is_grayscale = FALSE

    # Show or Clear panel handlers
    def showLeftPanel(self, img):
        self.panelLeft.configure(image=img)
        self.panelLeft.image = img

    def showRightPanel(self, img):
        self.panelRight.configure(image=img)
        self.panelRight.image = img

    def showBottomPanel(self):
        self.showButtonPanel()

    def showButtonPanel(self):
        buttonFrame = tk.Frame(self.root)
        buttonFrame.grid(row=2, column=0)

        clearEffBtn = tk.Button(buttonFrame, text="Очистить", command=self.clearEffect)
        clearEffBtn.grid(row=0, column=0, rowspan=3, padx=(0, 15))

        sobel_3x3_btn = tk.Button(buttonFrame, text="sobel_3x3_btn", command=self.sobel_3x3)
        sobel_3x3_btn.grid(row=0, column=1, sticky="ew")

        log_method = tk.Button(buttonFrame, text="log_method", command=self.laplacian_of_gaussian)
        log_method.grid(row=0, column=2, sticky="ew")

        sobel_5x5_btn = tk.Button(buttonFrame, text="sobel_5x5_btn", command=self.sobel_5x5)
        sobel_5x5_btn.grid(row=1, column=1, sticky="ew")

        GaussFilter2Btn = tk.Button(buttonFrame, text="has no func")
        GaussFilter2Btn.grid(row=2, column=2, sticky="ew")

        dog_method = tk.Button(buttonFrame, text="dog_method", command=self.difference_of_gaussian)
        dog_method.grid(row=1, column=2, sticky="ew")

        sobel_7x7_btn = tk.Button(buttonFrame, text="sobel_7x7_btn", command=self.sobel_7x7)
        sobel_7x7_btn.grid(row=2, column=1, sticky="ew")

        sobel_video_3x3 = tk.Button(buttonFrame, text="cv2_win_3x3", command=self.video_3x3)
        sobel_video_3x3.grid(row=0, column=3, sticky="ew", padx=(15, 0))

        sobel_video_5x5 = tk.Button(buttonFrame, text="cv2_win_5x5", command=self.video_5x5)
        sobel_video_5x5.grid(row=1, column=3, sticky="ew", padx=(15, 0))

        sobel_video_7x7 = tk.Button(buttonFrame, text="cv2_win_7x7", command=self.video_7x7)
        sobel_video_7x7.grid(row=2, column=3, sticky="ew", padx=(15, 0))

        log_video = tk.Button(buttonFrame, text="cv2_win_log", command=self.video_log)
        log_video.grid(row=0, column=4, sticky="ewns")

        dog_video = tk.Button(buttonFrame, text="cv2_win_dog", command=self.video_dog)
        dog_video.grid(row=1, column=4, sticky="ewns")

        origin_video = tk.Button(buttonFrame, text="cv2_win_source", command=self.video_origin)
        origin_video.grid(row=2, column=4, sticky="ewns")

        Brightness_CutBtn = tk.Button(buttonFrame, text="has no func")
        Brightness_CutBtn.grid(row=3, column=4, sticky="ewns")

        grayscaleBtn = tk.Button(buttonFrame, text="GRAY MODE", command=self.grayImage)
        grayscaleBtn.grid(row=0, column=5, sticky="ewns")

        showTextBtn = tk.Button(buttonFrame, text="Show errors", command=self.showTextPanel)
        showTextBtn.grid(row=1, column=5, sticky="ewns")

        # empty_slot = tk.Label(buttonFrame, text="123")
        # empty_slot.grid(row=0, column=5, sticky="ewns", rowspan=3)

    def showTextPanel(self):
        textFrame = tk.Frame(self.root)
        textFrame.grid(row=2, column=1)

        delta = tk.Label(textFrame, text="Delta: " + self.Delta())
        delta.grid(row=0, column=0, sticky="ewns")

        MSE = tk.Label(textFrame, text="MSE: " + self.MSE())
        MSE.grid(row=1, column=0, sticky="ewns")

        MSAD = tk.Label(textFrame, text="MSAD: " + self.MSAD())
        MSAD.grid(row=2, column=0, sticky="ewns")

    def clearLeftPanel(self):
        self.panelLeft.configure(image='')

    def clearRightPanel(self):
        self.panelRight.configure(image='')

    def clearHistogramCanvas(self):
        self.histCanvas.get_tk_widget().grid_remove()

    # Handle view mode changes
    def refreshImg(self):
        img = Image.fromarray(self.img)
        self.setSize(img)
        img = ImageTk.PhotoImage(img)
        self.showImage(img)

    def toImgMode(self):
        self.viewMode = 0
        self.showCurrentModeText()
        self.clearHistogramCanvas()
        self.panelRight.grid(row=1, column=1, ipadx=20, ipady=20)

        self.refreshImg()

    def toHistMode(self):
        self.viewMode = 1
        self.showCurrentModeText()
        self.clearRightPanel()
        self.showHistogramCanvas()

        self.refreshImg()

    def showCurrentModeText(self):
        if (self.currentMode != NULL):
            self.currentMode.destroy()

        if (self.viewMode == 0):
            self.currentMode = tk.Label(self.root, text="Лабораторная 2")
            self.currentMode.config(font=("Courier", 14))
            self.currentMode.grid(row=0, column=0, columnspan=2)
        elif (self.viewMode == 1):
            self.currentMode = tk.Label(self.root, text="Гистограмма")
            self.currentMode.config(font=("Courier", 14))
            self.currentMode.grid(row=0, column=0, columnspan=2)

    ################# Image Processing Operations #################
    def sobel_3x3(self):
        if self.is_grayscale == TRUE:
            sobel_img = copy.deepcopy(self.img)
            sobel_img = filters.sobel_filter_three_1(self.img, sobel_img)
            sobel = Image.fromarray(sobel_img)
            self.setImgArray(sobel)
            self.setSize(sobel)
            image_tk = ImageTk.PhotoImage(sobel)
            self.showImage(image_tk)
        else:
            tkinter.messagebox.showinfo(title="Image processing error", message="Your image is not gray")

    def sobel_5x5(self):
        if self.is_grayscale == TRUE:
            sobel_img = copy.deepcopy(self.img)
            sobel_img = filters.sobel_filter_five_1(self.img, sobel_img)
            sobel = Image.fromarray(sobel_img)
            self.setImgArray(sobel)
            self.setSize(sobel)
            image_tk = ImageTk.PhotoImage(sobel)
            self.showImage(image_tk)
        else:
            tkinter.messagebox.showinfo(title="Image processing error", message="Your image is not gray")

    def sobel_7x7(self):
        if self.is_grayscale == TRUE:
            sobel_img = copy.deepcopy(self.img)
            sobel_img = filters.sobel_filter_seven_1(self.img, sobel_img)
            sobel = Image.fromarray(sobel_img)
            self.setImgArray(sobel)
            self.setSize(sobel)
            image_tk = ImageTk.PhotoImage(sobel)
            self.showImage(image_tk)
        else:
            tkinter.messagebox.showinfo(title="Image processing error", message="Your image is not gray")

    def laplacian_of_gaussian(self):
        sigma_log = 2
        sigma_blur = 3
        if self.is_grayscale == TRUE:
            org = copy.deepcopy(self.img)
            copy_org = copy.deepcopy(self.img)
            blur = cv.GaussianBlur(self.img, (3, 3), sigma_blur)
            size = math.ceil(6 * pow(2, 0.5) * sigma_log)
            LoG = np.ndarray(shape=(size, size))
            kernel_rad = int(size / 2)

            log_image = clm.log(org, sigma_log, copy_org, LoG, kernel_rad, blur)
            log_image = Image.fromarray(log_image)
            self.setImgArray(log_image)
            self.setSize(log_image)
            image_tk = ImageTk.PhotoImage(log_image)
            self.showImage(image_tk)
        else:
            tkinter.messagebox.showinfo(title="Image processing error", message="Your image is not gray")


    def difference_of_gaussian(self):
        sigma = 1.3
        coefficient = 1.5
        if self.is_grayscale == TRUE:
            high = cv.GaussianBlur(self.img, (3, 3), coefficient * sigma)
            low = cv.GaussianBlur(self.img, (3, 3), sigma)

            dog_image = clm.DoG(self.img, high, low)
            dog_image = Image.fromarray(dog_image)
            self.setImgArray(dog_image)
            self.setSize(dog_image)
            image_tk = ImageTk.PhotoImage(dog_image)
            self.showImage(image_tk)
        else:
            tkinter.messagebox.showinfo(title="Image processing error", message="Your image is not gray")

    def video_3x3(self):
        cv.destroyAllWindows()
        cap = cv.VideoCapture("videoplayback.mp4")
        while(True):
            _, frame = cap.read()
            sobel3x3 = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            sobel_img = copy.deepcopy(sobel3x3)
            sobel_img = filters.sobel_filter_three_1(sobel3x3, sobel_img)
            cv.imshow("sobel_3x3", sobel_img)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

    def video_5x5(self):
        cv.destroyAllWindows()
        cap = cv.VideoCapture("videoplayback.mp4")
        while(True):
            _, frame = cap.read()
            sobel5x5 = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            sobel_img = copy.deepcopy(sobel5x5)
            sobel_img = filters.sobel_filter_five_1(sobel5x5, sobel_img)
            cv.imshow("sobel_5x5", sobel_img)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

    def video_7x7(self):
        cv.destroyAllWindows()
        cap = cv.VideoCapture("videoplayback.mp4")
        while(True):
            _, frame = cap.read()
            sobel7x7 = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            sobel_img = copy.deepcopy(sobel7x7)
            sobel_img = filters.sobel_filter_seven_1(sobel7x7, sobel_img)
            cv.imshow("sobel_7x7", sobel_img)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

    def video_log(self):
        cv.destroyAllWindows()
        cap = cv.VideoCapture("videoplayback.mp4")
        while(True):
            _, frame = cap.read()

            sigma_log = 2
            sigma_blur = 3

            gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            org = gray_frame
            copy_org = copy.deepcopy(gray_frame)

            blur = cv.GaussianBlur(org, (3, 3), sigma_blur)
            size = math.ceil(6 * pow(2, 0.5) * sigma_log)
            LoG = np.ndarray(shape=(size, size))
            kernel_rad = int(size / 2)

            log_image = clm.log(org, sigma_log, copy_org, LoG, kernel_rad, blur)

            cv.imshow("LoG", log_image)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

    def video_dog(self):
        cv.destroyAllWindows()
        cap = cv.VideoCapture("videoplayback.mp4")
        while(True):
            _, frame = cap.read()

            sigma = 1.3
            coefficient = 1.5

            gray_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

            high = cv.GaussianBlur(gray_frame, (3, 3), coefficient * sigma)
            low = cv.GaussianBlur(gray_frame, (3, 3), sigma)

            dog_image = clm.DoG(gray_frame, high, low)

            cv.imshow("DoG", dog_image)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

    def video_origin(self):
        cv.destroyAllWindows()
        cap = cv.VideoCapture("videoplayback.mp4")
        while(True):
            _, frame = cap.read()
            cv.imshow("source", frame)
            k = cv.waitKey(5) & 0xFF
            if k == 27:
                break

    def grayImage(self):
        self.clearRightPanel()
        self.handleGray(self.img)
        grayedImg = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)

        grayedImg = Image.fromarray(grayedImg)
        self.setImgArray(grayedImg)
        self.setSize(grayedImg)
        image_tk = ImageTk.PhotoImage(grayedImg)

        self.is_grayscale = TRUE
        self.showImage(image_tk)




    ################# Errors between two images #################

    def Delta(self):
        width = self.img.shape[1]
        height = self.img.shape[0]
        grayOri = cv.cvtColor(self.oriImg, cv.COLOR_BGR2GRAY)

        Delta = clm.delta(self.img, grayOri, height, width)

        return str(round(Delta, 2))

    def MSE(self):
        width = self.img.shape[1]
        height = self.img.shape[0]
        grayOri = cv.cvtColor(self.oriImg, cv.COLOR_BGR2GRAY)

        MSE = clm.mse(self.img, grayOri, height, width)

        return str(round(MSE, 2))

    def MSAD(self):
        width = self.img.shape[1]
        height = self.img.shape[0]
        grayOri = cv.cvtColor(self.oriImg, cv.COLOR_BGR2GRAY)

        MSAD = clm.msad(self.img, grayOri, height, width)

        return str(round(MSAD, 2))

    ################# Image Histogram #################
    def showHistogramCanvas(self):
        self.panelRight.grid_remove()
        self.clearHistogramCanvas()

        self.histFigure = plt.Figure(figsize=(5.2, 5.2), dpi=100)

        if (self.is_grayscale == FALSE):
            # RED канал
            redChannel = self.histFigure.add_subplot(221)
            redChannel.plot(cv.calcHist([self.img], [0], None, [256], [0, 256]), color="red")
            redChannel.title.set_text("По красному каналу")

            # Green Канал
            greenChannel = self.histFigure.add_subplot(222)
            greenChannel.plot(cv.calcHist([self.img], [1], None, [256], [0, 256]), color="green")
            greenChannel.title.set_text("По зеленому каналу")

            # Blue Канал
            blueChannel = self.histFigure.add_subplot(223)
            blueChannel.plot(cv.calcHist([self.img], [2], None, [256], [0, 256]), color="blue")
            blueChannel.title.set_text("По синему каналу")

            # График профиля яркости
            any_y = 228
            x_arr = np.array([])
            for i in range(self.img.shape[1]):
                x_arr = np.append(x_arr, sum(self.img[any_y][i]) / 3)
            profile_brightness = self.histFigure.add_subplot(224)
            profile_brightness.plot(x_arr, color="yellow")
            profile_brightness.title.set_text(f"Профиль яркости в строке {any_y}")

            # Assign figure to canvas to show in main window
            self.histFigure.suptitle("Гистограмма")
            self.histCanvas = FigureCanvasTkAgg(self.histFigure, self.root)
            self.histCanvas.get_tk_widget().grid(row=1, column=1)
        else:
            grayHist = self.histFigure.add_subplot(111)
            grayHist.plot(cv.calcHist([self.img], [0], None, [256], [0, 256]), color="gray")
            self.histFigure.suptitle("Гистограмма оттенка серого")
            self.histCanvas = FigureCanvasTkAgg(self.histFigure, self.root)
            self.histCanvas.get_tk_widget().grid(row=1, column=1)

    def donothing(self):
        pass


Main(Tk())
