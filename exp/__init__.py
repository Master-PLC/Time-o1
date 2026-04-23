from .exp_long_term_forecasting import Exp_Long_Term_Forecast
from .exp_short_term_forecasting import Exp_Short_Term_Forecast
from .exp_long_term_forecasting_meta_ml3 import Exp_Long_Term_Forecast_META_ML3
from .exp_long_term_forecasting_trans import Exp_Long_Term_Forecast_Trans
from .exp_short_term_forecasting_trans import Exp_Short_Term_Forecast_Trans

EXP_DICT = {
    'long_term_forecast': Exp_Long_Term_Forecast,
    'long_term_forecast_meta_ml3': Exp_Long_Term_Forecast_META_ML3,
    'short_term_forecast': Exp_Short_Term_Forecast,
    'long_term_forecast_trans': Exp_Long_Term_Forecast_Trans,
    'short_term_forecast_trans': Exp_Short_Term_Forecast_Trans,
}