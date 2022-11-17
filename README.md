# Master's Project: [ALMA Lensing Cluster Survey](https://ui.adsabs.harvard.edu/abs/2019asrc.confE..64K/abstract)
This project makes use of optical/NIR data from HST and *Spitzer* along with ALMA 1.2 mm data to study galaxies which are magnified by lensing clusters. This unique feature allows us to explore those faint distant galaxies.
[`a370_skyimage.ipynb`](https://github.com/hunyjm12/Master_Project_ALCS/blob/main/a370_skyimage.ipynb) shows how I matched optical/NIR sources with ALMA-detected sources, so as to conduct further analysis.

[`MatchAndPlot.ipynb`](https://github.com/hunyjm12/Master_Project_ALCS/blob/main/MatchAndPlot.py) carries out match across all 33 lensing clusters and displays their properties deduced from optical/NIR SED fitting. Examples of results are in https://github.com/hunyjm12/Master_Project_ALCS/tree/main/Examples-of-Output

[`mbb.ipynb`](https://github.com/hunyjm12/Master_Project_ALCS/blob/main/mbb.ipynb) calculates $L_{\mathrm{IR}}$ and $M_{\mathrm{dust}}$ by rescaling the modified blackbody to the observed 1.2 mm ALMA data.

[`SfrCalculate.ipynb`](https://github.com/hunyjm12/Master_Project_ALCS/blob/main/SfrCalculate.ipynb) calculates $SFR_{\mathrm{UV+IR}}$ and displays different properties in 2-D planes.

[My Master's thesis](https://drive.google.com/file/d/14LDebriG7b2aEqj90HhS4Usiixjc1wkU/view?usp=sharing)
