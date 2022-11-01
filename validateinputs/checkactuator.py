from validateinputs import registerplugin
from validateinputs import CheckStatus as status
import os.path
import OpenFASTutil

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
            turbtype = default_type if 'Actuator_type' not in tdict else tdict['Actuator_type']
            fstfile  = ''
            turbtype = turbtype[0] if isinstance(turbtype, list) else turbtype
            if turbtype not in ['TurbineFastLine', 'TurbineFastDisk']:
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

            # Now check for other things in the input file
            if checkstatus['result'] == status.PASS:
                # Check CompInflow
                check = Check_Actuator_FST_CompInflow.check(app, fstfile, subname=turb)
                for c in check: checklist.append(c)

                check = Check_Actuator_FST_Aerodyn.check(app, fstfile, subname=turb)
                for c in check: checklist.append(c)

        return checklist

# Don't register this one, it's called by Check_Actuator_FSTfile
class Check_Actuator_FST_CompInflow(): 
    """
    Check CompInflow
    """
    name = "FST CompInflow"

    @classmethod
    def check(self, app, fstfile, subname=''):
        checkstatus = {'subname':subname}  
        # Get compinflow from fstfile
        CompInflow = int(OpenFASTutil.getVarFromFST(fstfile, 'CompInflow'))
        if CompInflow == 2:
            checkstatus['result']  = status.PASS
            checkstatus['mesg']    = 'CompInflow OK'
        else:
            checkstatus['result']  = status.FAIL
            checkstatus['mesg']    = 'CompInflow=%i, should be 2.'%CompInflow
        return [checkstatus]
            
# Don't register this one, it's called by Check_Actuator_FSTfile
class Check_Actuator_FST_Aerodyn(): 
    """
    Check Aerodyn inputs
    """
    name = "FST Aerodyn"

    @classmethod
    def check(self, app, fstfile, subname=''):
        allchecks   = []

        checkaerodyn = {'subname':subname}  
        CompAero = int(OpenFASTutil.getVarFromFST(fstfile, 'CompAero'))
        if CompAero in [1,2]:
            # Check to make sure AeroFile exists
            AeroFile = OpenFASTutil.getVarFromFST(fstfile, 'AeroFile').strip('"')
            AeroFileWPath = os.path.join(os.path.dirname(fstfile), AeroFile)
            if os.path.isfile(AeroFileWPath): 
                checkaerodyn['result']  = status.PASS
                checkaerodyn['mesg']    = '[%s] exists'%AeroFileWPath
            else:
                checkaerodyn['result']  = status.FAIL
                checkaerodyn['mesg']    = \
                 'AeroFile=[%s] does not exist'%AeroFileWPath
        else:
            checkaerodyn['result']  = status.SKIP
            checkaerodyn['mesg']    = 'CompAero=%i... skipping Aerodyn'%CompAero
        allchecks.append(checkaerodyn)

        # Now check for things in the AerodynFile
        if checkaerodyn['result']  == status.PASS:
            # Check WakeMod
            WakeMod = int(OpenFASTutil.getVarFromFST(AeroFileWPath, 'WakeMod'))
            checkwakemod = {'subname':subname}  
            if WakeMod == 0:
                checkwakemod['result']  = status.PASS
                checkwakemod['mesg']    = 'WakeMod=%i OK'%WakeMod
            else:
                checkwakemod['result']  = status.FAIL
                checkwakemod['mesg']    = 'WakeMod=%i, should be 0'%WakeMod
            pass
            allchecks.append(checkwakemod)

            # Check Density
            checkdensity = {'subname':subname}
            AirDens  = OpenFASTutil.getDensity(fstfile, verbose=False)
            incflo_density = app.inputvars['density'].getval()
            if abs(AirDens - incflo_density) > 1.0E-6:
                checkdensity['result'] = status.WARN
                checkdensity['mesg']   = 'AirDens=%f, does not match incflo.density=%f'%(AirDens, incflo_density)
            else:
                checkdensity['result'] = status.PASS
                checkdensity['mesg']   = 'AirDens=%f, matches incflo.density=%f'%(AirDens, incflo_density)
            allchecks.append(checkdensity)

        return allchecks
