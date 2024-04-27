from validateinputs import registerplugin
from validateinputs import CheckStatus as status
from validateinputs import setcheckstatus
import os

"""
See README.md for details on the structure of classes here
"""


@registerplugin
class Check_max_level(): 
    name = "max_level"

    def check(self, app):
        max_level = app.inputvars['max_level'].getval()

        checkstatus                = {}   # Dict containing return status
        checkstatus['subname']     = ''   # Additional name info
        if max_level >= 0:
            checkstatus['result']  = status.PASS  
            checkstatus['mesg']    = 'max_level = %i >= 0'%max_level
        else:
            checkstatus['result']  = status.FAIL
            checkstatus['mesg']    = 'max_level = %i < 0'%max_level            
        return [checkstatus]              # Must be a list of dicts

@registerplugin
class Check_dt_cfl(): 
    name = "dt & CFL"

    def check(self, app):
        dt  = app.inputvars['fixed_dt'].getval()
        cfl = app.inputvars['cfl'].getval()

        checkstatus                = {}   # Dict containing return status
        checkstatus['subname']     = ''   # Additional name info
        if (dt<0.0) and (cfl<0.0):
            checkstatus['result']  = status.FAIL
            checkstatus['mesg']    = 'Both dt = %f<0 and cfl=%f < 0'%(dt, cfl)
        else:
            checkstatus['result']  = status.PASS  
            checkstatus['mesg']    = 'DT and CFL OK'
        return [checkstatus]              # Must be a list of dicts

@registerplugin
class Check_restart_dir():
    name = "restart dir"

    def check(self, app):
        checkstatus                = {}   # Dict containing return status
        checkstatus['subname']     = ''   # Additional name info

        restartdir = app.inputvars['restart_file'].getval()
        if restartdir is None:
            setcheckstatus(checkstatus, status.SKIP, 'No restart file specified')
        else:
            # Check to make sure that the restart dir exists
            if os.path.exists(restartdir):
                setcheckstatus(checkstatus, status.PASS, 'Restart directory %s exists'%restartdir)
            else:
                setcheckstatus(checkstatus, status.FAIL, 'Restart directory %s does not exist'%restartdir)
        return [checkstatus]

@registerplugin
class Check_bndry_dir():
    name = "boundary plane dir"

    def check(self, app):
        checkstatus                = {}   # Dict containing return status
        checkstatus['subname']     = ''   # Additional name info

        iomode = int(app.inputvars['ABL_bndry_io_mode'].getval())
        if iomode == 1:
            bndrydir = app.inputvars['ABL_bndry_file'].getval()
            # Check to make sure that the boundary data dir exists
            if os.path.exists(bndrydir):
                setcheckstatus(checkstatus, status.PASS, 'Restart directory %s exists'%bndrydir)
            else:
                setcheckstatus(checkstatus, status.FAIL, 'Restart directory %s does not exist'%bndrydir)
        else:
            setcheckstatus(checkstatus, status.SKIP, 'ABL.bndry_io_mode is not 1')
        return [checkstatus]
