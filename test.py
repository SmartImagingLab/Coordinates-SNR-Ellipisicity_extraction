# -*- coding: utf-8 -*-
from LSEExtraction import PSFExtractor

if __name__ == "__main__":
    #Input the FITS path or auto retrieve the program path
    Path = input("please input the fits file's path:")
    #Input the output txt file path or automatically save to the project path
    Save = input("please input data saving path:")
    #Input the size of the side lengths of a square mesh. The side lengths of fits is multiples of mesh.
    mesh = input("please input a mesh size:")

    sextractor = PSFExtractor(Path, Save, mesh)
    # Get the coordinates of stars in the fits
    sextractor.get_location()
    # Get the SNR of the fits
    sextractor.get_snr()
    # Get the Ellipisicity of stars in the fits
    sextractor.get_ellipisicity()
