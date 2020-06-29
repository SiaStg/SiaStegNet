import numpy as np
import matlab.engine

def PySUNIWARD(cover, payload, eng): #cover: nparray
    stego = eng.S_UNIWARD(matlab.double(cover.tolist()), payload)
    return np.asarray(matlab.double(stego), dtype='uint8') 
