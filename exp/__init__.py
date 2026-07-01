import sys

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast


EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'short_term_forecast': Exp_Short_Term_Forecast,
}