from . import utils
from . import simulate
from . import dataset
from .config import SUPPORTED_SAMPLE_TYPES, SAMPLE_RATE, USE_SIMULATED_DATA


generate_power_series = utils.generate_power_series
find_nearest = utils.find_nearest
random_float = utils.random_float
show_sample = utils.show_sample
add_noise = simulate.add_noise
interpolate = simulate.interpolate
parse_sample = dataset.parse_sample
if USE_SIMULATED_DATA:
    get_sample = dataset.get_sample_fake
    generate_dataset = dataset.generate_dataset_fake
else:
    get_sample = dataset.get_sample_real
    generate_dataset = dataset.generate_dataset_real
