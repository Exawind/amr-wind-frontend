# wavenumber_spectra

Calculates 2D wavenumber spectra in x and y
## Inputs: 
```
  name                : An arbitrary name (Required)
  ncfile              : NetCDF file of horizontal planes (Required)
  group               : Group name in netcdf file (Required)
  trange              : Which times to average over (Required)
  num_bins            : How many wavenumber bins to use for spectra (Optional, Default: 50)
  type                : What type of spectra to compute: 'energy', 'horiz', 'vertical', 'kol'. Default is all. (Optional, Default: ['energy', 'horiz', 'vertical', 'kol'])
  varnames            : Variables to extract from the netcdf file (Optional, Default: ['velocityx', 'velocityy', 'velocityz'])
  iplanes             : i-index of planes to postprocess (Optional, Default: None)
  csvfile             : Filename to save spectra to (Optional, Default: '')
  xaxis               : Which axis to use on the abscissa (Optional, Default: 'x')
  yaxis               : Which axis to use on the ordinate (Optional, Default: 'y')
  C_kol               : Kolmogorov constant to use for theoretical spectra (Optional, Default: 1.5)
  diss_rate           : Dissipation rate for theoretical spectra (Optional, Default: 1.0)
  remove_endpoint_x   : Remove one endpoint in x before FFT if periodic signal is sampled twice at endpoints. (Optional, Default: False)
  remove_endpoint_y   : Remove one endpoint in y before FFT if periodic signal is sampled twice at endpoints. (Optional, Default: False)
```

## Actions: 

## Notes

Specifically, the Fourier transform of the two-point velocity correlation $R_{ij}(oldsymbol{r},t) = \langle u_i(oldsymbol{x},t) u_j(oldsymbol{x}+oldsymbol{r},t) angle$ is computed from the FFT of the sampled AMR-Wind velocity data at a given height, $z$, as


```math
\hat{R}_{ij}(oldsymbol{r},t) = \langle \hat{u}^*_i(oldsymbol{\kappa},t) \hat{u}_j(oldsymbol{\kappa},t) angle
.
```


Here, $oldsymbol{x} = (x,y)$ is a 2D horizontal vector, and $oldsymbol{r} = (r_x,r_y)$ is a 2D separation vector.
The velocity spectrum tensor, $\Phi_{ij}(oldsymbol{\kappa},t)$, for a 2D wavenumber vector $oldsymbol{\kappa} = (\kappa_x,\kappa_y)$, is then computed as


```math
\Phi_{ij}(oldsymbol{\kappa},t) \equiv \sum_{oldsymbol{\kappa'}} \delta(oldsymbol{\kappa} - oldsymbol{\kappa}') \hat{R}_{ij}(oldsymbol{\kappa}',t) pprox \hat{R}_{ij}(oldsymbol{\kappa},t)/(\Delta oldsymbol{\kappa}),
```


such that


```math
R_{ij}(oldsymbol{r},t) = \iint \Phi_{ij}(oldsymbol{\kappa},t) e^{i oldsymbol{\kappa} \cdot oldsymbol{r}} d oldsymbol{\kappa}.
```


The two dimensional spectra are then computed as surface integrals in 2D wavenumber space. Specifically, we denote the circle in wavenumber space, centered at the origin, with radius $\kappa = |oldsymbol{\kappa}|$ as $\mathcal{S}(\kappa)$. Then the integration over the surface of this circle is approximated as


```math
\oint f(oldsymbol{\kappa}) d \mathcal{S}(\kappa) pprox
rac{2 \pi \kappa}{N} \sum^N_{ |\kappa' - \kappa| < d\kappa } f(oldsymbol{\kappa}')
,
```


where $N$ is the total number of points in the summation for each wavenumber magnitude.
This is applied to different components of the velocity spectrum tensor to compute the energy, horizontal, and vertical spectra as:

- Energy spectra:


```math
E = \oint  rac{1}{2} \Phi_{ii}(oldsymbol{\kappa},t) d\mathcal{S}(\kappa)
```


- Horizontal spectra:


```math
E = \oint  rac{1}{2} \left[ \Phi_{11}(oldsymbol{\kappa},t) + \Phi_{22}(oldsymbol{\kappa},t)  ight] d\mathcal{S}(\kappa)
```


- Vertical spectra:


```math
E = \oint \Phi_{33}(oldsymbol{\kappa},t) d\mathcal{S}(\kappa)
```


## Example

```yaml
    wavenumber_spectra:
        name: Spectra_027
        ncfile: XYdomain_027_30000.nc
        group: Farm_XYdomain027
        csvfile: E_spectra_Z027.csv
        trange:  [15000, 20000]
        iplanes: 0
```
    
