from importlib import reload
from .mesher import Mesh
from .data import Data
from .fitter import Fit
from .fasteval import FEMatrix

reload_modules = True
if reload_modules:
    import morphic.mesher
    reload(mesher)
    import morphic.interpolator
    reload(interpolator)
    from morphic.mesher import Mesh
    import morphic.fitter
    reload(fitter)
    from morphic.fitter import Fit

