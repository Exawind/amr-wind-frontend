#!/usr/bin/env python

import sys
sys.path.insert(1, './tkyamlgui')
import numpy as np
from functools import partial
import tkyamlgui as tkyg
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

class MyApp(tkyg.App, object):
    def __init__(self, *args, **kwargs):
        super(MyApp, self).__init__(*args, **kwargs)
        t   = np.arange(0, 3, .01)
        self.fig.clf()
        self.fig.add_subplot(111).plot(t, t**2)

        # Add an exit button
        exitbutton = Tk.Button(master=self.notebook.tab('Tab 2'), 
                               text="Quit", command=self.quit)
        exitbutton.grid(row=10, column=0, padx=5, sticky='w')
        self.formatgridrows()

if __name__ == "__main__":
    MyApp(configyaml='config.yaml', title='AMR-Wind input').mainloop()
