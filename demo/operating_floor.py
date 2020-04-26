# -*- coding: utf-8 -*-
from LSEExtraction import PSFExtractor

if __name__ == "__main__":
    #the FITS path 
    Path = r'C:\Users\孙永阳\Desktop\demo'
    #the output txt file path
    Save = r'C:\Users\孙永阳\Desktop\demo'
    #Input the size of the side lengths of a square mesh. The side lengths of fits is multiples of mesh.
    mesh = 100

    sextractor = PSFExtractor(Path, Save, mesh)
    # Get the coordinates of stars in the fits
    sextractor.get_location()
    # Get the SNR of the fits
    sextractor.get_snr()
    # Get the Ellipisicity of stars in the fits
    sextractor.get_ellipisicity()
