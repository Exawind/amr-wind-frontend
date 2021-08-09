#!/usr/bin/env python

amrwindfedir = '../../'
import sys, os
sys.path.insert(1, amrwindfedir)

# Load the libraries
import matplotlib.pyplot    as plt
import amrwind_frontend as amrwind

case = amrwind.MyApp.init_nogui()
case.setAMRWindInput('time.stop_time', 20000.0)
case.setAMRWindInput('time.max_step',  40000)

case.setAMRWindInput('time.fixed_dt',  0.5)
case.setAMRWindInput('time.checkpoint_interval',  2000)

case.setAMRWindInput('incflo.physics', ['ABL'])
case.setAMRWindInput('incflo.verbose', 3)
case.setAMRWindInput('io.check_file', 'chk')

case.setAMRWindInput('incflo.use_godunov', True)
case.setAMRWindInput('incflo.godunov_type', 'weno')

case.setAMRWindInput('turbulence.model',    ['OneEqKsgsM84'])
case.setAMRWindInput('TKE.source_terms',    ['KsgsM84Src'])

# I'm lazy, just do this in a string
tols = """
nodal_proj.mg_rtol                       = 1e-06               
nodal_proj.mg_atol                       = 1e-12               
mac_proj.mg_rtol                         = 1e-06               
mac_proj.mg_atol                         = 1e-12               
diffusion.mg_rtol                        = 1e-06               
diffusion.mg_atol                        = 1e-12               
temperature_diffusion.mg_rtol            = 1e-10               
temperature_diffusion.mg_atol            = 1e-13               
"""
case.loadAMRWindInput(tols, string=True)

case.setAMRWindInput('transport.viscosity', 1.8375e-05)

case.setAMRWindInput('geometry.prob_lo', [ 0.0, 0.0, 0.0 ])
case.setAMRWindInput('geometry.prob_hi', [1536.0, 1536.0, 1920.0])
case.setAMRWindInput('amr.n_cell',       [128, 128, 160])

case.setAMRWindInput('is_periodicx', True)
case.setAMRWindInput('is_periodicy', True)
case.setAMRWindInput('is_periodicz', False)

case.setAMRWindInput('zlo.type',              'wall_model')          
case.setAMRWindInput('zlo.temperature_type',  'wall_model')          
case.setAMRWindInput('zlo.tke_type',          'zero_gradient')       
case.setAMRWindInput('zhi.type',              'slip_wall')           
case.setAMRWindInput('zhi.temperature_type',  'fixed_gradient')      
case.setAMRWindInput('zhi.temperature',       0.000974025974) 

case.setAMRWindInput('ICNS.source_terms',     ['ABLForcing','BoussinesqBuoyancy', 'CoriolisForcing'])

case.setAMRWindInput('ABL.stats_output_frequency',   1)                   
case.setAMRWindInput('ABL.stats_output_format',      'netcdf')

case.setAMRWindInput('incflo.velocity', [4.70059422901, 3.93463008353, 0.0])
case.setAMRWindInput('ABLForcing.abl_forcing_height',   57.19)
case.setAMRWindInput('ABL.kappa',                       0.4) 

case.setAMRWindInput('ABL.normal_direction',      2)
case.setAMRWindInput('ABL.surface_roughness_z0',  0.0001)
case.setAMRWindInput('ABL.reference_temperature', 288.15)
case.setAMRWindInput('ABL.surface_temp_rate',     0.0)
case.setAMRWindInput('ABL.surface_temp_flux',     0.0122096146646)

case.setAMRWindInput('ABL.mo_beta_m',             16.0)
case.setAMRWindInput('ABL.mo_gamma_m',            5.0)
case.setAMRWindInput('ABL.mo_gamma_h',            5.0)
case.setAMRWindInput('ABL.random_gauss_mean',     0.0)
case.setAMRWindInput('ABL.random_gauss_var',      1.0)

case.setAMRWindInput('CoriolisForcing.latitude',  55.49)
case.setAMRWindInput('BoussinesqBuoyancy.reference_temperature', 288.15) 

case.setAMRWindInput('ABL.temperature_heights', '1050.0 1150.0 1920.0')
case.setAMRWindInput('ABL.temperature_values',  '288.15 296.15 296.9')

# This is a case where we don't want to use the defaults in amrwind-frontend
case.setAMRWindInput('ABL.perturb_ref_height', None)
case.setAMRWindInput('ABL.Uperiods', None)
case.setAMRWindInput('ABL.Vperiods', None)
case.setAMRWindInput('ABL.deltaU',   None)
case.setAMRWindInput('ABL.deltaV',   None)
case.setAMRWindInput('ABL.theta_amplitude',   None)
case.setAMRWindInput('ABL.cutoff_height',   None)

case.setAMRWindInput('time.plot_interval', 2000)
case.setAMRWindInput('incflo.post_processing', ['sampling'])            

case.setAMRWindInput('sampling.output_frequency', 100)                 
case.setAMRWindInput('sampling.fields',           ['velocity', 'temperature'])

sampleplane = case.get_default_samplingdict()
# Modify the geometry
sampleplane['sampling_name']         = 'p_hub'
sampleplane['sampling_type']         = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = [129, 129]
sampleplane['sampling_p_origin']     = [0, 0, 0]
sampleplane['sampling_p_axis1']      = [1536, 0, 0]
sampleplane['sampling_p_axis2']      = [0, 1536, 0]
sampleplane['sampling_p_normal']     = [0, 0, 1]
sampleplane['sampling_p_offsets']    = '17        28.5      41        57        77        90'
case.add_sampling(sampleplane)

sampleplane = case.get_default_samplingdict()
sampleplane['sampling_name']         = 'xbc'
sampleplane['sampling_type']         = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = [257, 161]
sampleplane['sampling_p_origin']     = [0, 0, 0]
sampleplane['sampling_p_axis1']      = [0, 1536, 0]
sampleplane['sampling_p_axis2']      = [0, 0, 1920]
sampleplane['sampling_p_normal']     = [1, 0, 0]
sampleplane['sampling_p_offsets']    = '0.0 1536'
case.add_sampling(sampleplane)

sampleplane = case.get_default_samplingdict()
sampleplane['sampling_name']         = 'ybc'
sampleplane['sampling_type']         = 'PlaneSampler'
sampleplane['sampling_p_num_points'] = [257, 161]
sampleplane['sampling_p_origin']     = [0, 0, 0]
sampleplane['sampling_p_axis1']      = [1536, 0, 0]
sampleplane['sampling_p_axis2']      = [0, 0, 1920]
sampleplane['sampling_p_normal']     = [0, 1, 0]
sampleplane['sampling_p_offsets']    = '0.0 1536'
case.add_sampling(sampleplane)

#print(case.inputvars['physics'].inputtype)
print(case.writeAMRWindInput(''))

