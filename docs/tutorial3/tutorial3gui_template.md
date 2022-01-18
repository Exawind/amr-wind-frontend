
# Tutorial 3: Setting up a farm calculation

<!-- Variables within braces (between {{ and }}) will be replaced by
the python script make_tutorial3_gui_markdownimages.py -->

## Introduction

This tutorial will do something...

## Set some wind properties

WS = {WS}  m/s
WDir =  {WDir} degrees

| Input                   | Value      |
| ---                     | ---        |
| Use speed & dir instead | {useWSDir} |
| Wind speed              | {WS}       |
| Wind direction          | {WDir}     |

![{img_ABL_settings}]({img_ABL_settings})

Hit the **[Calc WS/WDir]** button, and it should fill in the values of
the Wind Vector appropriately.

## Create wind farm layout and domain

The first thing we'll do is to set the wind farm layout and set the
location of each of the turbines. Click on the **Farm** tab, and hit
**[show]** next to **Wind farm turbine layout**.

Copy and paste this input into the **CSV Contents** window.
```
{turbinescsv}
```

If you have all of the turbine inputs in a separate CSV file, you can
also load them by hitting **[Choose file]**, then selecting the file,
and hitting **[Load/Reload]**.

The other inputs in the
| Input                    | value              |
| ---                      | ---                |
| Farm domain size (X,Y,Z) | {domainsize}       |
| Background mesh size [m] | {backgrounddeltax} |

![{img_farm_turbine_layout}]({img_farm_turbine_layout})

If you have questions about what each of the columns in the CSV input
mean, hit the **[?]** button.  You should see a help window that
resembles:

{turbinecsv_help}

Press the **[Preview]** button under **Actions**.  It should generate
a turbine and domain layout image which resembles something like this:  

![{img_turbine_layout_preview}]({img_turbine_layout_preview})

There should be two turbines labeled T1 and T0 in the middle of a 1 km
x 1 km domain.

If you want to adjust any turbine positions or orientations, go back
and edit the CSV input above.  Then press the **[Create turbines]**
button to actually create the turbines in the simulation.

To verify that the turbines are actually created, click the
**Turbines** tab and then look at the **Turbine List** under **Add
turbines here**.  You should see that both T0 and T1 are included in
the list:

![{img_farm_turbine_created}]({img_farm_turbine_created})


## Create wind farm refinement zones  

Now around each turbine, we'll add some refinement zones so the
turbine rotors can be resolved.  Click back on the **Farm** tab, and
hit **[show]** next to **Farm refinement zones**.

```
{refinementcsv}
```

![{img_farm_refinementspec}]({img_farm_refinementspec})

If you have questions about what each of the CSV columns are supposed
to mean, hit the **[?]** button.  You should see a help window with
the following information:

{refinecsv_help}

Press the **[Create refinements]** button to ceate the refinement
zones.  Then, to see what the refinement zones look like, open the
plot domain dialog on the menu bar: **Plot**-->**Plot domain**.  You
should see that there 4 refinement zones present, and the two
previously created turbines also present.  

![{img_plotDomainWin_refinezone}]({img_plotDomainWin_refinezone})

Hit **[Select all]** for both the refinement zones and turbines, then
press **[Plot Domain]**.  You should see both turbines plotted, and
each refinement level around the turbines shown.  Everything should be
oriented so that it points into the wind:

![{img_plotDomainFig_refineturbine}]({img_plotDomainFig_refineturbine})

## Create sampling planes

```
{samplingcsv}
```

![{img_farm_samplingspec}]({img_farm_samplingspec})

{samplingcsv_help}

![{img_plotDomainWin_samplingzone}]({img_plotDomainWin_samplingzone})

![{img_plotDomainFig_refineturbinesampling}]({img_plotDomainFig_refineturbinesampling})

## See the output 

To see what the AMR-Wind input file would look like, you can go to the
menu bar, select **Run** --> **Preview Input File**, to see the
preview window (you can also just hit **File** --> **Save input file
as** to save it to a file).

The input file should look similar to:  
<details>
  <summary>[Expand input file]</summary>
<pre>
{amrwindinput1}
</pre>
</details>

## Set up a wind sweep

The above instructions were to set up a single case with one wind
speed ({WS} m/s) and one wind direction ({WDir}) degrees.  Now we'll
show what happens when you want to vary these to run a parameter
sweep.  Hit **[show]** next to **Run parameter sweep**, and put in the
following inputs:

| Input                 | Value              |
| ---                   | ---                |
| Wind speeds           | {sweep_windspeeds} |
| Wind directions       | {sweep_winddirs}   |
| Case prefix           | {caseprefix}       |
| New dir for each case | {usenewdirs}       |
| Logfile               | {logfile}          |

The set up should be similar to: 

![{img_farm_runsweepspec}]({img_farm_runsweepspec})

Then hit the **[Create cases]** button.  Four input files should be generated:
```
Tutorial3_Case_0.inp
Tutorial3_Case_1.inp
Tutorial3_Case_2.inp
Tutorial3_Case_3.inp
```
as well as the yaml log file `{logfile}` which looks like this:  
```yaml
{logfileoutput}
```