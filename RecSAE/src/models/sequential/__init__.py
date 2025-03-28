from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]

# Explicit imports for new SAE models with improvements
from .SASRec_SAE_Attention import SASRec_SAE_Attention
from .SASRec_SAE_Contrastive import SASRec_SAE_Contrastive
from .SASRec_SAE_Hierarchical import SASRec_SAE_Hierarchical
