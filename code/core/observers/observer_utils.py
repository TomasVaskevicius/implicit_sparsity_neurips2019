import numpy as np

def _aggregate_numeric_results(observers, attribute):
    """ This helper function can be used by observers, which during each
    notify() call saves a number to a list in a given attribute name.
    This function then concatenates all results into a numpy array. """
    all_outputs = []
    for observer in observers:
        observer_list = getattr(observer, attribute)
        if len(observer_list) is 0:
            # If empty by default return -1.
            observer_list = [-1]
        all_outputs.append(np.stack(observer_list))
    all_outputs_array = np.stack(all_outputs)

    if all_outputs_array.ndim == 2:
        # Each observation is a single number and hence the
        # all_outputs_array is two dimensional now. To be consistent in all
        # cases we add a singleton dimension in this case.
        all_outputs_array = np.expand_dims(all_outputs_array, axis = 2)

    # The returned array will have shape:
    # (n_observers, n_observations_per_observer) + (observation_dimension,).
    return all_outputs_array
