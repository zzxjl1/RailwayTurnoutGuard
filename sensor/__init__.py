from . import utils
from . import simulate
from .config import SUPPORTED_SAMPLE_TYPES, SAMPLE_RATE, USE_SIMULATED_DATA


generate_power_series = utils.generate_power_series
find_nearest = utils.find_nearest
random_float = utils.random_float
show_sample = utils.show_sample
add_noise = simulate.add_noise
interpolate = simulate.interpolate
if USE_SIMULATED_DATA:
    generate_sample = simulate.generate_sample
else:
    pass