import importlib

from breast_metadata.study import *
from breast_metadata.configurations import *
from breast_metadata.scan import *
from breast_metadata.image import *
try:
    from breast_metadata.sitk_tools import *
except:
    pass
try:
    from breast_metadata.pyvista_tools import *
except:
    pass

# Import modules using OpenCMISS-Iron.
try:
    opencmiss_iron_spec = importlib.util.find_spec("SimpleITK")
except:
    print(
        "SimpleITK module not installed, skipping import of "
        "breast_metadata/sitk_tools.py")
else:
    from breast_metadata.sitk_tools import *
