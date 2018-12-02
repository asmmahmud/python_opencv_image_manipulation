import tkinter

try:
    from PIL import Image, ImageFont, ImageDraw, ImageTk
except ImportError:
    import Image

loadedImg = Image.open("../resources/img/nid.jpg")

# use a truetype font
font = ImageFont.truetype("../resources/img/HindSiliguri-SemiBold.ttf", 75)

draw = ImageDraw.Draw(loadedImg)
draw.text((10, 25), "মাহমুদুল Hasan", font=font, fill=(50, 50, 50, 128))

window = tkinter.Tk()

canvas = tkinter.Canvas(window, width=992, height=600)
canvas.pack()

photo = ImageTk.PhotoImage(loadedImg)

canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

window.mainloop()

# PIL.ImageGrab.grab
