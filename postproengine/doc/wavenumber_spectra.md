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

The Fourier transform of the two-point velocity correlation $R_{ij}(\boldsymbol{r},t) = \langle u_i(\boldsymbol{x},t) u_j(\boldsymbol{x}+\boldsymbol{r},t) \rangle$ is computed from the FFT of the sampled AMR-Wind velocity data at a given height, $z$, as


```math
\hat{R}_{ij}(\boldsymbol{r},t) = \langle \hat{u}^*_i(\boldsymbol{\kappa},t) \hat{u}_j(\boldsymbol{\kappa},t) \rangle
.
```


Here, $\boldsymbol{x} = (x,y)$ is a 2D horizontal vector, and $\boldsymbol{r} = (r_x,r_y)$ is a 2D separation vector.
The velocity spectrum tensor, $\Phi_{ij}(\boldsymbol{\kappa},t)$, for a 2D wavenumber vector $\boldsymbol{\kappa} = (\kappa_x,\kappa_y)$, is then computed as


```math
\Phi_{ij}(\boldsymbol{\kappa},t) \equiv \sum_{\boldsymbol{\kappa'}} \delta(\boldsymbol{\kappa} - \boldsymbol{\kappa}') \hat{R}_{ij}(\boldsymbol{\kappa}',t) \approx \hat{R}_{ij}(\boldsymbol{\kappa},t)/(\Delta \boldsymbol{\kappa}),
```


such that


```math
R_{ij}(\boldsymbol{r},t) = \iint \Phi_{ij}(\boldsymbol{\kappa},t) e^{i \boldsymbol{\kappa} \cdot \boldsymbol{r}} d \boldsymbol{\kappa}.
```


The two dimensional spectra are then computed as surface integrals in 2D wavenumber space. Specifically, we denote the circle in wavenumber space, centered at the origin, with radius $\kappa = |\boldsymbol{\kappa}|$ as $\mathcal{S}(\kappa)$. Then the integration over the surface of this circle is approximated as


```math
\oint f(\boldsymbol{\kappa}) d \mathcal{S}(\kappa) \approx
\frac{2 \pi \kappa}{N} \sum^N_{ |\kappa' - \kappa| < d\kappa } f(\boldsymbol{\kappa}')
,
```


where $N$ is the total number of points in the summation for each wavenumber magnitude.
This is applied to different components of the velocity spectrum tensor to compute the energy, horizontal, and vertical spectra as:

- Energy spectra:


```math
E = \oint  \frac{1}{2} \Phi_{ii}(\boldsymbol{\kappa},t) d\mathcal{S}(\kappa)
```


- Horizontal spectra:


```math
E = \oint  \frac{1}{2} \left[ \Phi_{11}(\boldsymbol{\kappa},t) + \Phi_{22}(\boldsymbol{\kappa},t)  \right] d\mathcal{S}(\kappa)
```


- Vertical spectra:


```math
E = \oint \Phi_{33}(\boldsymbol{\kappa},t) d\mathcal{S}(\kappa)
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
    
