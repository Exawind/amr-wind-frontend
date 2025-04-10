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
        default_type   = default_type[0] if len(default_type) > 1 else 'TurbineFastLine'
        default_turbD  = app.inputvars['Actuator_%s_rotor_diameter'%default_type].getval()
        default_hh     = app.inputvars['Actuator_%s_hub_height'%default_type].getval()
        default_density= app.inputvars['Actuator_%s_density'%default_type].getval()
        #default_turbD  = app.inputvars['Actuator_TurbineFastLine_rotor_diameter'].getval()
        #default_hh     = app.inputvars['Actuator_TurbineFastLine_hub_height'].getval()
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
                # Check AMR density
                amrdensity = default_type if 'Actuator_density' not in tdict else tdict['Actuator_density']
                check = Check_Actuator_AMR_density.check(app, amrdensity, subname=turb)
                for c in check: checklist.append(c)

                # Check CompInflow
                check = Check_Actuator_FST_CompInflow.check(app, fstfile, subname=turb)
                for c in check: checklist.append(c)

                check = Check_Actuator_FST_Aerodyn.check(app, fstfile, subname=turb)
                for c in check: checklist.append(c)

        return checklist

# Don't register this one, it's called by Check_Actuator_FSTfile
class Check_Actuator_AMR_density():
    """
    Check density
    """
    name = "AMR density"

    @classmethod
    def check(self, app, amrdensity, subname=''):
        checkstatus = {'subname':subname}
        incflo_density = app.inputvars['density'].getval()
        if amrdensity is None:
            checkstatus['result'] = status.FAIL
            checkstatus['mesg']   = 'Actuator density=%s, does not match incflo.density=%f'%(repr(amrdensity), incflo_density)
        elif abs(amrdensity - incflo_density) > 1.0E-6:
            checkstatus['result'] = status.FAIL
            checkstatus['mesg']   = 'Actuator density=%f, does not match incflo.density=%f'%(amrdensity, incflo_density)
        else:
            checkstatus['result'] = status.PASS
            checkstatus['mesg']   = 'Actuator density=%f, matches incflo.density=%f'%(amrdensity, incflo_density)
        return [checkstatus]


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
        eps  = 1.0E-6
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
            WakeMod  = OpenFASTutil.getVarFromFST(AeroFileWPath, 'WakeMod')
            Wake_Mod = OpenFASTutil.getVarFromFST(AeroFileWPath, 'Wake_Mod')
            WakeModFinal = int(WakeMod) if WakeMod is not None else int(Wake_Mod)
            checkwakemod = {'subname':subname}  
            if WakeModFinal == 0:
                checkwakemod['result']  = status.PASS
                checkwakemod['mesg']    = 'WakeMod=%i OK'%WakeModFinal
            else:
                checkwakemod['result']  = status.FAIL
                checkwakemod['mesg']    = 'WakeMod=%i, should be 0'%WakeModFinal
            pass
            allchecks.append(checkwakemod)

            # Check Density
            checkdensity = {'subname':subname}
            AirDens  = OpenFASTutil.getDensity(fstfile, verbose=False)
            incflo_density = app.inputvars['density'].getval()
            if abs(AirDens - incflo_density) > eps:
                checkdensity['result'] = status.WARN
                checkdensity['mesg']   = 'AirDens=%f, does not match incflo.density=%f'%(AirDens, incflo_density)
            else:
                checkdensity['result'] = status.PASS
                checkdensity['mesg']   = 'AirDens=%f, matches incflo.density=%f'%(AirDens, incflo_density)
            allchecks.append(checkdensity)

            # Check DISCON density (if needed)
            checkdensityDISCON = {'subname':subname}
            CompServo  = int(OpenFASTutil.getVarFromFST(fstfile, 'CompServo'))
            ServoFile  = OpenFASTutil.getVarFromFST(fstfile, 'ServoFile').strip('"')
            checkdensityDISCON['result'] = status.SKIP
            checkdensityDISCON['mesg']   = 'Skipping DISCON density check'
            if (CompServo == 1):
                # Check the servo file controller
                ServoFileWPath = os.path.join(os.path.dirname(fstfile), ServoFile)
                PCMode = int(OpenFASTutil.getVarFromFST(ServoFileWPath, 'PCMode'))
                if PCMode == 5:
                    DLL_InFile = OpenFASTutil.getVarFromFST(ServoFileWPath, 'DLL_InFile').strip('"')
                    DLL_ProcName = OpenFASTutil.getVarFromFST(ServoFileWPath, 'DLL_ProcName').strip('"').upper()
                    if DLL_ProcName == "DISCON":
                        DISCONFileWPath = os.path.join(os.path.dirname(fstfile), DLL_InFile)
                        WE_RhoAir = float(OpenFASTutil.getVarFromDISCON(DISCONFileWPath, 'WE_RhoAir'))
                        print(f'GOT WE_RhoAir = {WE_RhoAir}')
                        if abs(WE_RhoAir - AirDens) + abs(WE_RhoAir - incflo_density) > 2.0*eps:
                            checkdensityDISCON['result'] = status.WARN
                            checkdensityDISCON['mesg']   = 'WE_RhoAir=%f in DISCON does not match AirDens=%f and incflo.density=%f'%(WE_RhoAir, AirDens, incflo_density)
                        else:
                            checkdensityDISCON['result'] = status.PASS
                            checkdensityDISCON['mesg']   = 'WE_RhoAir=%f in DISCON matches AirDens=%f and incflo.density=%f'%(WE_RhoAir, AirDens, incflo_density)
            allchecks.append(checkdensityDISCON)
        return allchecks
