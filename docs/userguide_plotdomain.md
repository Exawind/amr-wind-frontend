# Plotting the domain

## Through the interactive gui

1. Start `amrwind_frontend` and load the input file with the domain.
   You can specify the input file on the command line:
   ```bash
   $ ./amrwind_frontend.py sample.inp
   ```
   or launch the `amrwind_frontend.py` first, then select `File` ->
   `Import AMR-Wind file` to choose the file.

2.  Select `Plot`->`Plot domain` from the menu bar.  This should bring
    up a plot domain window like below:
	
    ![snapshot](amrwind_frontend_plotdomain_popup.png)
	
3.  Choose the features to plot.  For instance, in the picture above,
    the `p_f` sampling plane is chosen, which is the small blue plane
    in the corner.  Also, the static refinement box `s1` was also
    plotted in the picture, which consists of two levels.

    ![snapshot](amrwind_frontend_plotdomain_example.png)
	
	
