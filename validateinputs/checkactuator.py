from validateinputs import registerplugin
from validateinputs import CheckStatus as status
import os.path

"""
See README.md for details on the structure of classes here
"""

""" --- HERE IS THE TEMPLATE ---
@registerplugin
class PluginA(): 
    name   = "Plugin A"
    #active = False
    def check(self, app):
        return [{'subname':'', 'result':status.SKIP, 'mesg':''}]
"""

@registerplugin
class Check_Actuator_Physics(): 
    name = "Actuator physics"

    def check(self, app):
        incflo_physics  = app.inputvars['physics'].getval()
        ActuatorForcing = app.inputvars['ActuatorForcing'].getval()

        checkstatus                = {}   # Dict containing return status
        checkstatus['subname']     = ''   # Additional name info

        if (ActuatorForcing and ('Actuator' in incflo_physics)) or \
           ((not ActuatorForcing) and ('Actuator' not in incflo_physics)):
            checkstatus['result']  = status.PASS
            checkstatus['mesg']    = 'incflo.physics and ICNS.source_terms OK for Actuators'

        if (ActuatorForcing and ('Actuator' not in incflo_physics)) or \
           ((not ActuatorForcing) and ('Actuator' in incflo_physics)):
            checkstatus['result']  = status.FAIL
            checkstatus['mesg']    = 'incflo.physics must have Actuator and ICNS.source_terms must have ActuatorForcing'
        return [checkstatus]
            
@registerplugin
class Check_Actuator_FSTfile(): 
    name = "Actuator FST"

    def check(self, app):
        ActuatorForcing = app.inputvars['ActuatorForcing'].getval()
        checklist = []   # list of all checkstatuses

        # Skip if no actuator
        if (not ActuatorForcing):
            checkstatus                = {}   # Dict containing return status
            checkstatus['subname']     = ''   # Additional name info
            checkstatus['result']  = status.SKIP
            checkstatus['mesg']    = 'No Actuators'
            return [checkstatus]

        # Go through actuators and check for FST files if necessary
        allturbines  = app.listboxpopupwindict['listboxactuator']
        allturbtags  = allturbines.getitemlist()
        keystr       = lambda n, d1, d2: d2.name

        # Get the defaults
        default_type   = app.inputvars['Actuator_default_type'].getval()
        default_turbD  = app.inputvars['Actuator_TurbineFastLine_rotor_diameter'].getval()
        default_hh     = app.inputvars['Actuator_TurbineFastLine_hub_height'].getval()
        for turb in allturbtags:
            checkstatus                = {'mesg':''}   
            checkstatus['subname']     = turb
            tdict = allturbines.dumpdict('AMR-Wind',
                                         subset=[turb], keyfunc=keystr)
            turbtype = default_type if 'actuator_individual_type' not in tdict else tdict['actuator_individual_type']
            if turbtype[0] not in ['TurbineFastLine', 'TurbineFastDisk']:
                checkstatus['result']      = status.SKIP
                checkstatus['mesg']        = 'Not OpenFAST'                
            else:
                fstfile  = tdict['Actuator_openfast_input_file']
                if os.path.isfile(fstfile): 
                    checkstatus['result']      = status.PASS
                    checkstatus['mesg']        = '[%s] exists'%fstfile
                else:
                    checkstatus['result']      = status.FAIL
                    checkstatus['mesg']        = '[%s] does NOT exist'%fstfile 
            checklist.append(checkstatus)
        
        return checklist
