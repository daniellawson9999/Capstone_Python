import wx

print(wx.version())
app=wx.App()  # Need to create an App instance before doing anything
app.__init__()
dc=wx.Display.GetCount()
print(dc)
#e(0)
displays = (wx.Display(i) for i in range(wx.Display.GetCount()))
sizes = [display.GetGeometry().GetSize() for display in displays]

for (i,s) in enumerate(sizes):
    print("Monitor{} size is {}".format(i,s))   
screen = wx.ScreenDC()
#pprint(dir(screen))
size = screen.GetSize()

print("Width = {}".format(size[0]))
print("Heigh = {}".format(size[1]))

width=size[0]
height=size[1]
x,y,w,h =putty_rect

bmp = wx.Bitmap(w,h)
mem = wx.MemoryDC(bmp)

for i in range(98):
    if 1:
        #1-st display:

        #pprint(putty_rect)
        #e(0)

        mem.Blit(-x,-y,w+x,h+y, screen, 0,0)

    if 0:
        #2-nd display:
        mem.Blit(0, 0, x,y, screen, width,0)
    #e(0)

    if 0:
        #3-rd display:
        mem.Blit(0, 0, width, height, screen, width*2,0)

    bmp.SaveFile(os.path.join(home,"image_%s.jpg" % i), wx.BITMAP_TYPE_JPEG)    
    print (i)
    sleep(0.2)
del mem