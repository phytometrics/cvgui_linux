import tkinter as tk
import tkinter.ttk as ttk
import numpy as np
from PIL import Image, ImageTk
import os
import skimage
import cv2


def convert_rgb2gray(image):
    print("aaaa",end=" ")
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image

def test(image):
    pass

def create_module_func():
    module_list = [
        convert_rgb2gray,
        test
    ]

    module_dict = {s.__name__: s for s in module_list}
    return module_dict

def get_modules():
    modules = [
        convert_rgb2gray,
        test
    ]
    return [x.__name__ for x in modules]


class App(tk.Frame):
    def __init__(self, master=tk.Tk(), camera=None):
        super().__init__(master)
        #self.image: np.NDarray
        self.master.geometry("1000x1000")
        self.settings_frame = tk.Frame(self.master, bg="black")
        self.settings_frame.pack(side=tk.TOP, anchor="n", fill="x")

        self.pause = False

        self.module_dict = create_module_func()


        self.module_selection_cb = ttk.Combobox(self.settings_frame,
                                                value=get_modules(),
                                                state="readonly")

        self.module_selection_cb.pack(side=tk.TOP)
        self.module_selection_cb.bind("<<ComboboxSelected>>", self.change_combobox)
        self.image_frame = tk.Canvas(self.master, bg="white")
        self.image_frame.pack(side=tk.TOP, anchor="n",expand=True,fill="both")

        filename = os.path.join(skimage.data_dir, 'chelsea.png')
        self.image = cv2.imread(filename)

        self.update()

    def change_combobox(self,event):
        print("change",flush=True)
        self.pause = True
        new_func_string = event.widget.get()
        print(self.image_analysis_module)
        self.image_analysis_module = self.module_dict[new_func_string]
        print(self.image_analysis_module)

        self.pause = False


    def image_analysis_module(self,image):
        return image

    def update(self):


        # self.image = np.random.rand(200, 200, 3) * 255
        # self.image = self.image.astype(np.uint8)
        if not self.pause:
            self.image = self.image_analysis_module(self.image)
            input_size = [self.image_frame.winfo_width(), self.image_frame.winfo_height()]
            self.width = self.master.winfo_screenwidth() // 2
            self.height = self.master.winfo_screenheight()
            self.canvas_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.canvas_image = cv2.resize(self.canvas_image, dsize=tuple(input_size))
            self.canvas_image = Image.fromarray(self.canvas_image)
            self.canvas_image = ImageTk.PhotoImage(image=self.canvas_image)
            self.image_frame.create_image(0, 0, image=self.canvas_image, anchor="nw")
            self.master.after(1, self.update)
        else:
            print("pause")

if __name__ == "__main__":
    app = App()
    app.mainloop()