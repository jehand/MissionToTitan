#This function, spkez, uses spice to get the ephemeras data of the objects

import spiceypy as spice
import numpy as np

#get ephemeras data from a given time array
def load_ephemeras(target, times, frame, observer ):
    if type( target ) == str:
        return np.array(spice.spkezr( target, times, frame, 'NONE', observer )[0] )

    else:
        return np.array(spice.spkez( target, times, frame, 'NONE', observer ) [0] )

