#  SPDX-FileCopyrightText: 2023 EasyScience contributors  <core@easyscience.software>
#  SPDX-License-Identifier: BSD-3-Clause
#  © 2021-2023 Contributors to the EasyScience project <https://github.com/easyScience/EasyScience

__author__ = 'github.com/wardsimon'
__version__ = '0.1.0'

import weakref
from typing import Any
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple
from typing import TypeVar
from typing import Union

import numpy as np

# import pint_xarray
import xarray as xr

from easyscience import ureg
from easyscience.fitting import FitResults

T_ = TypeVar('T_')


@xr.register_dataset_accessor('easyscience')
class EasyScienceDatasetAccessor:
    """
        Accessor to extend an xarray DataSet to EasyScience. These functions can be accessed by `obj.EasyScience.func`.
        The objective for this class accessor is to facilitate:

    - Creation of nd datasets by making and assigning axes and data more accessible
    - To add and keep track of errors in the form of sigma for datasets.py
    - To perform fitting on one or more data arrays in the dataset simultaneously.
    - To facilitate dask computation if required.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        This is called whenever you access obj.EasyScience, hence the attributes in the obj should only be written if they
        have not been previously instantiated.

        :param xarray_obj: DataSet which is called by obj.EasyScience
        :type xarray_obj: xarray.Dataset
        """

        self._obj = xarray_obj
        self._core_object = None
        self.__error_mapper = {}
        self.sigma_label_prefix = 's_'
        if self._obj.attrs.get('name', None) is None:
            self._obj.attrs['name'] = ''
        if self._obj.attrs.get('description', None) is None:
            self._obj.attrs['description'] = ''
        if self._obj.attrs.get('url', None) is None:
            self._obj.attrs['url'] = ''
        if self._obj.attrs.get('units', None) is None:
            self._obj.attrs['units'] = {}

    @property
    def name(self) -> str:
        """
        Get the common name of the DataSet.

        :return: Common name of the DataSet
        :rtype: str
        """
        return self._obj.attrs['name']

    @name.setter
    def name(self, new_name: str):
        """
        Set the common name of the DataSet i.e could be experiment name...

        :param new_name: Common name of the DataSet
        :type new_name: str
        :return: None
        :rtype: None
        """
        self._obj.attrs['name'] = new_name

    @property
    def description(self) -> str:
        """
        Get a description of the DataSet

        :return: Description of the DataSet
        :rtype: str
        """
        return self._obj.attrs['description']

    @description.setter
    def description(self, new_description: str):
        """
        Set the description of the DataSet

        :param new_description: Description of the DataSet
        :type new_description: str
        :return: None
        :rtype: None
        """
        self._obj.attrs['description'] = new_description

    @property
    def url(self) -> str:
        """
        Get the url of the DataSet

        :return: URL of the DataSet (empty if no URL)
        :rtype: str
        """
        return self._obj.attrs['url']

    @url.setter
    def url(self, new_url: str):
        """
        Set the URL of the DataSet. This may be a DOI.

        :param new_url: New URL/DOI of the DataSet
        :type new_url: str
        :return:None
        :rtype: None
        """
        self._obj.attrs['url'] = new_url

    @property
    def core_object(self):
        """
        Get the core object associated to a DataSet. Note that this is called from a weakref. If the EasyScience obj is
        garbage collected, None will be returned.

        :return: EasyScience object associated with the DataSet
        :rtype: Any
        """
        if self._core_object is None:
            return None
        return self._core_object()

    @core_object.setter
    def core_object(self, new_core_object: Any):
        """
        Associate an EasyScience object to a DataSet.

        :param new_core_object: EasyScience object to be associated to the DataSet
        :type new_core_object: Any
        :return: None
        :rtype: None
        """
        self._core_object = weakref.ref(new_core_object)

    def add_coordinate(
        self,
        coordinate_name: str,
        coordinate_values: Union[List[T_], np.ndarray],
        unit: str = '',
    ):
        """
        Add a coordinate to the DataSet. This can be then be assigned to one or more DataArrays.

        :param coordinate_name: Name of the coordinate e.g. `x`
        :type coordinate_name: str
        :param coordinate_values: Points for the coordinates
        :type coordinate_values: Union[List[T_], numpy.ndarray]
        :param unit: Unit associated with the coordinate
        :type unit: str
        :return: None
        :rtype: None
        """
        self._obj.coords[coordinate_name] = coordinate_values
        self._obj.attrs['units'][coordinate_name] = ureg.Unit(unit)

    def remove_coordinate(self, coordinate_name: str):
        """
        Remove a coordinate from the DataSet. Note that this will not remove the coordinate from DataArrays which have
        already used the it!

        :param coordinate_name: Name of the coordinate to be removed
        :type coordinate_name: str
        :return: None
        :rtype: None
        """
        del self._obj.coords[coordinate_name]
        del self._obj.attrs['units'][coordinate_name]

    def add_variable(
        self,
        variable_name,
        variable_coordinates: Union[str, List[str]],
        variable_values: Union[List[T_], np.ndarray],
        variable_sigma: Union[List[T_], np.ndarray] = None,
        unit: str = '',
        auto_sigma: bool = False,
    ):
        """
        Create a DataArray from known coordinates and data, assign it to the dataset under a given name. Variances can
        be calculated assuming gaussian distribution to 1 sigma.

        :param variable_name: Name of the DataArray which will be created and added to the dataset
        :type variable_name: str
        :param variable_coordinates: List of coordinates used in the supplied data array.
        :type variable_coordinates: str, List[str]
        :param variable_values: Numpy or list of data which will be assigned to the DataArray
        :type variable_values: Union[numpy.ndarray, list]
        :param variable_sigma: If the sigmas of the dataset are known, they can be supplied here.
        :type variable_sigma: Union[numpy.ndarray, list]
        :param unit: Unit associated with the DataArray
        :type unit: str
        :param auto_sigma: Should the sigma DataArray be automatically calculated assuming gaussian probability?
        :type auto_sigma: bool
        :return: None
        :rtype: None
        """

        # Check if a user has supplied a coordinate as a string. Make it a list of strings
        if isinstance(variable_coordinates, str):
            variable_coordinates = [variable_coordinates]

        # The variable_coordinates can be any iterable object. Though we would assume list/tuple
        if not isinstance(variable_coordinates, Iterable):
            raise ValueError('The variable coordinates must be a list of strings')

        # Check to see if the user want to assign a coordinate which does not exist yet.
        known_keys = self._obj.coords.keys()
        for dimension in variable_coordinates:
            if dimension not in known_keys:
                raise ValueError(f'The supplied coordinate `{dimension}` must first be defined.')

        # Create  the dataset.
        self._obj[variable_name] = (variable_coordinates, variable_values)

        # Deal with sigmas
        if variable_sigma is not None:
            # CASE 1, user has supplied sigmas
            if isinstance(variable_sigma, Callable):
                # CASE 1-1, The sigmas are created by some kind of generator
                self.sigma_generator(variable_name, variable_sigma)
            elif isinstance(variable_sigma, np.ndarray):
                # CASE 1-2, The sigmas are a numpy arrays
                self.sigma_attach(variable_name, variable_sigma)
            elif isinstance(variable_sigma, list):
                # CASE 1-3, We have been given a list. Make it a numpy array
                self.sigma_attach(variable_name, np.array(variable_sigma))
            else:
                raise ValueError('User supplied sigmas must be of the form; Callable fn, numpy array, list')
        else:
            # CASE 2, No sigmas have been supplied.
            if auto_sigma:
                # CASE 2-1, Automatically generate the sigmas using gaussian probability
                self.sigma_generator(variable_name)

        # Set units for the newly created DataArray
        self._obj.attrs['units'][variable_name] = ureg.Unit(unit)
        # If a sigma has been attached, attempt to work out the units.
        if unit and variable_sigma is None and auto_sigma:
            self._obj.attrs['units'][self.sigma_label_prefix + variable_name] = ureg.Unit(unit + ' ** 0.5')
        else:
            if auto_sigma:
                self._obj.attrs['units'][self.sigma_label_prefix + variable_name] = ureg.Unit('')

    def remove_variable(self, variable_name: str):
        """
        Remove a DataArray from the DataSet by supplied name.

        :param variable_name: Name of DataArray to be removed
        :type variable_name: str
        :return: None
        :rtype: None
        """
        del self._obj[variable_name]

    def sigma_generator(
        self,
        variable_label: str,
        sigma_func: Callable = lambda x: np.sqrt(np.abs(x)),
        label_prefix: str = None,
    ):
        """
        Generate sigmas off of a DataArray based on a function.

        :param variable_label: Name of the DataArray to perform the calculation on
        :type variable_label: str
        :param sigma_func: Function to generate the sigmas. Must be of the form f(x) and return an array of the same shape as the input. Default sqrt(\\|x\\|)
        :type sigma_func: Callable
        :param label_prefix: What prefix should be used to designate a sigma DataArray from a data DataArray
        :type label_prefix: str
        :return: None
        :rtype: None
        """  # noqa: E501
        sigma_values = sigma_func(self._obj[variable_label])
        self.sigma_attach(variable_label, sigma_values, label_prefix)

    def sigma_attach(
        self,
        variable_label: str,
        sigma_values: Union[List[T_], np.ndarray, xr.DataArray],
        label_prefix: str = None,
    ):
        """
        Attach an array of sigmas to the DataSet.

        :param variable_label: Name of the DataArray to perform the calculation on
        :type variable_label: str
        :param sigma_values: Array of sigmas in list, numpy or DataArray form
        :type sigma_values: Union[List[T_], numpy.ndarray, xarray.DataArray]
        :param label_prefix: What prefix should be used to designate a sigma DataArray from a data DataArray
        :type label_prefix: str
        :return: None
        :rtype: None
        """
        # Use the default sigma prefix if not defined.
        if label_prefix is None:
            label_prefix = self.sigma_label_prefix

        # Form the label for the new DataArray
        sigma_label = label_prefix + variable_label

        # Map the original DataArray to the new sigma DataArray
        self.__error_mapper[variable_label] = sigma_label
        # Assign the sigma DataArray to the DataSet
        if not isinstance(sigma_values, xr.DataArray):
            self._obj[sigma_label] = (
                list(self._obj[variable_label].coords.keys()),
                sigma_values,
            )
        else:
            self._obj[sigma_label] = sigma_values

    def generate_points(self, coordinates: List[str]) -> xr.DataArray:
        """
                Generate an expanded DataArray of points which corresponds to broadcasted dimensions (`all_x`) which have been
                concatenated along the second axis (`fit_dim`).

                :param coordinates: List of coordinate names to broadcast and concatenate along
                :type coordinates: List[str]
                :return: Broadcasted and concatenated coordinates
                :rtype: xarray.DataArray

        .. code-block:: python

             x = [1, 2], y = [3, 4]
             d = xr.DataArray()
             d.EasyScience.add_coordinate('x', x)
             d.EasyScience.add_coordinate('y', y)
             points = d.EasyScience.generate_points(['x', 'y'])
             print(points)
        """

        coords = [self._obj.coords[da] for da in coordinates]
        c_array = []
        n_array = []
        for da in xr.broadcast(*coords):
            c_array.append(da)
            n_array.append(da.name)

        f = xr.concat(c_array, dim='fit_dim')
        f = f.stack(all_x=n_array)
        return f

    def fit(
        self,
        fitter,
        data_arrays: list,
        *args,
        dask: str = 'forbidden',
        fit_kwargs: dict = None,
        fn_kwargs: dict = None,
        vectorized: bool = False,
        **kwargs,
    ) -> List[FitResults]:
        """
        Perform a fit on one or more DataArrays. This fit utilises a given fitter from `EasyScience.fitting.Fitter`, though
        there are a few differences to a standard EasyScience fit. In particular, key-word arguments to control the
        optimisation algorithm go in the `fit_kwargs` dictionary, fit function key-word arguments go in the `fn_kwargs`
        and given key-word arguments control the `xarray.apply_ufunc` function.

        :param fitter: Fitting object which controls the fitting
        :type fitter: EasyScience.fitting.Fitter
        :param args: Arguments to go to the fit function
        :type args: Any
        :param dask: Dask control string. See `xarray.apply_ufunc` documentation
        :type dask: str
        :param fit_kwargs: Dictionary of key-word arguments to be supplied to the Fitting control
        :type fit_kwargs: dict
        :param fn_kwargs: Dictionary of key-words to be supplied to the fit function
        :type fn_kwargs: dict
        :param vectorized: Should the fit function be given dependents in a single object or split
        :type vectorized: bool
        :param kwargs: Key-word arguments for `xarray.apply_ufunc`. See `xarray.apply_ufunc` documentation
        :type kwargs: Any
        :return: Results of the fit
        :rtype: List[FitResults]
        """

        if fn_kwargs is None:
            fn_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        if not isinstance(data_arrays, (list, tuple)):
            data_arrays = [data_arrays]

        # In this case we are only fitting 1 dataset
        if len(data_arrays) == 1:
            variable_label = data_arrays[0]
            dataset = self._obj[variable_label]
            if self.__error_mapper.get(variable_label, False):
                # Pull out any sigmas and send them to the fitter.
                temp = self._obj[self.__error_mapper[variable_label]]
                temp[xr.ufuncs.isnan(temp)] = 1e5
                fit_kwargs['weights'] = temp
            # Perform a standard DataArray fit.
            return dataset.EasyScience.fit(
                fitter,
                *args,
                fit_kwargs=fit_kwargs,
                fn_kwargs=fn_kwargs,
                dask=dask,
                vectorize=vectorized,
                **kwargs,
            )
        else:
            # In this case we are fitting multiple datasets to the same fn!
            bdim_f = [self._obj[p].EasyScience.fit_prep(fitter.fit_function) for p in data_arrays]
            dim_names = [
                list(self._obj[p].dims.keys()) if isinstance(self._obj[p].dims, dict) else self._obj[p].dims
                for p in data_arrays
            ]
            bdims = [bdim[0] for bdim in bdim_f]
            fs = [bdim[1] for bdim in bdim_f]
            old_fit_func = fitter.fit_function

            fn_array = []
            y_list = []
            for _idx, d in enumerate(bdims):
                dims = self._obj[data_arrays[_idx]].dims
                if isinstance(dims, dict):
                    dims = list(dims.keys())

                def local_fit_func(x, *args, idx=None, **kwargs):
                    kwargs['vectorize'] = vectorized
                    res = xr.apply_ufunc(
                        fs[idx],
                        *bdims[idx],
                        *args,
                        dask=dask,
                        kwargs=fn_kwargs,
                        **kwargs,
                    )
                    if dask != 'forbidden':
                        res.compute()
                    return res.stack(all_x=dim_names[idx])

                y_list.append(self._obj[data_arrays[_idx]].stack(all_x=dims))
                fn_array.append(local_fit_func)

            def fit_func(x, *args, **kwargs):
                res = []
                for idx in range(len(fn_array)):
                    res.append(fn_array[idx](x, *args, idx=idx, **kwargs))
                return xr.DataArray(np.concatenate(res, axis=0), coords={'all_x': x}, dims='all_x')

            fitter.initialize(fitter.fit_object, fit_func)
            try:
                if fit_kwargs.get('weights', None) is not None:
                    del fit_kwargs['weights']
                x = xr.DataArray(np.arange(np.sum([y.size for y in y_list])), dims='all_x')
                y = xr.DataArray(np.concatenate(y_list, axis=0), coords={'all_x': x}, dims='all_x')
                f_res = fitter.fit(x, y, **fit_kwargs)
                f_res = check_sanity_multiple(f_res, [self._obj[p] for p in data_arrays])
            finally:
                fitter.fit_function = old_fit_func
            return f_res


@xr.register_dataarray_accessor('EasyScience')
class EasyScienceDataarrayAccessor:
    """
    Accessor to extend an xarray DataArray to EasyScience. These functions can be accessed by `obj.EasyScience.func`.

    """

    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj
        self._core_object = None
        self.sigma_label_prefix = 's_'
        if self._obj.attrs.get('computation', None) is None:
            self._obj.attrs['computation'] = {
                'precompute_func': None,
                'compute_func': None,
                'postcompute_func': None,
            }

    def __empty_functional(self) -> Callable:
        def outer():
            def empty_fn(input, *args, **kwargs):
                return input

            return empty_fn

        class wrapper:
            def __init__(obj):
                obj.obj = self
                obj.data = {}
                obj.fn = outer()

            def __call__(self, *args, **kwargs):
                return self.fn(*args, **kwargs)

        return wrapper()

    @property
    def core_object(self):
        """
        Get the core object associated to a DataArray. Note that this is called from a weakref. If the EasyScience obj is
        garbage collected, None will be returned.

        :return: EasyScience object associated with the DataArray
        :rtype: Any
        """
        if self._core_object is None:
            return None
        return self._core_object()

    @core_object.setter
    def core_object(self, new_core_object: Any):
        """
        Set the core object associated to a dataset

        :param new_core_object: EasyScience object to be associated with the DataArray
        :type new_core_object: Any
        :return: None
        :rtype: None
        """
        self._core_object = weakref.ref(new_core_object)

    @property
    def compute_func(self) -> Callable:
        """
        Get the computational function which will be executed during a fit

        :return: Computational function applied to the DataArray
        :rtype: Callable
        """
        result = self._obj.attrs['computation']['compute_func']
        if result is None:
            result = self.__empty_functional()
        return result

    @compute_func.setter
    def compute_func(self, new_computational_fn: Callable):
        """
        Set the computational function which is called during a fit

        :param new_computational_fn: Computational function applied to the DataArray
        :type new_computational_fn: Callable
        :return: None
        :rtype: None
        """
        self._obj.attrs['computation']['compute_func'] = new_computational_fn

    @property
    def precompute_func(self) -> Callable:
        """
        Get the pre-computational function which will be executed before a fit

        :return: Computational function applied to the DataArray before fitting
        :rtype: Callable
        """
        result = self._obj.attrs['computation']['precompute_func']
        if result is None:
            result = self.__empty_functional()
        return result

    @precompute_func.setter
    def precompute_func(self, new_computational_fn: Callable):
        """
        Set the computational function which is called before a fit

        :param new_computational_fn: Computational function applied to the DataArray before fitting
        :type new_computational_fn: Callable
        :return: None
        :rtype: None
        """
        self._obj.attrs['computation']['precompute_func'] = new_computational_fn

    @property
    def postcompute_func(self) -> Callable:
        """
        Get the post-computational function which will be executed after a fit

        :return: Computational function applied to the DataArray after fitting
        :rtype: Callable
        """
        result = self._obj.attrs['computation']['postcompute_func']
        if result is None:
            result = self.__empty_functional()
        return result

    @postcompute_func.setter
    def postcompute_func(self, new_computational_fn: Callable):
        """
        Set the computational function which is called after a fit

        :param new_computational_fn: Computational function applied to the DataArray after fitting
        :type new_computational_fn: Callable
        :return: None
        :rtype: None
        """
        self._obj.attrs['computation']['postcompute_func'] = new_computational_fn

    def fit_prep(self, func_in: Callable, bdims=None, dask_chunks=None) -> Tuple[xr.DataArray, Callable]:
        """
        Generate broadcasted coordinates for fitting and reform the fitting function into one which can handle xarrays.

        :param func_in: Function to be wrapped and made xarray fitting compatible.
        :type func_in: Callable
        :param bdims: Optional precomputed broadcasted dimensions.
        :type bdims: xarray.DataArray
        :param dask_chunks: How to split the broadcasted dimensions for dask.
        :type dask_chunks: Tuple[int..]
        :return: Tuple of broadcasted fit arrays and wrapped fit function.
        :rtype: xarray.DataArray, Callable
        """

        if bdims is None:
            coords = [self._obj.coords[da].transpose() for da in self._obj.dims]
            bdims = xr.broadcast(*coords)
        self._obj.attrs['computation']['compute_func'] = func_in

        def func(x, *args, vectorize: bool = False, **kwargs):
            old_shape = x.shape
            if not vectorize:
                xs = [x_new.flatten() for x_new in [x, *args] if isinstance(x_new, np.ndarray)]
                x_new = np.column_stack(xs)
                if len(x_new.shape) > 1 and x_new.shape[1] == 1:
                    x_new = x_new.reshape((-1))
                result = self.compute_func(x_new, **kwargs)
            else:
                result = self.compute_func(
                    *[d for d in [x, args] if isinstance(d, np.ndarray)],
                    *[d for d in args if not isinstance(d, np.ndarray)],
                    **kwargs,
                )
            if isinstance(result, np.ndarray):
                result = result.reshape(old_shape)
            result = self.postcompute_func(result)
            return result

        return bdims, func

    def generate_points(self) -> xr.DataArray:
        """
        Generate an expanded DataArray of points which corresponds to broadcasted dimensions (`all_x`) which have been
        concatenated along the second axis (`fit_dim`).

        :return: Broadcasted and concatenated coordinates
        :rtype: xarray.DataArray
        """

        coords = [self._obj.coords[da] for da in self._obj.dims]
        c_array = []
        n_array = []
        for da in xr.broadcast(*coords):
            c_array.append(da)
            n_array.append(da.name)

        f = xr.concat(c_array, dim='fit_dim')
        f = f.stack(all_x=n_array)
        return f

    def fit(
        self,
        fitter,
        *args,
        fit_kwargs: dict = None,
        fn_kwargs: dict = None,
        vectorize: bool = False,
        dask: str = 'forbidden',
        **kwargs,
    ) -> FitResults:
        """
        Perform a fit on the given DataArray. This fit utilises a given fitter from `EasyScience.fitting.Fitter`, though
        there are a few differences to a standard EasyScience fit. In particular, key-word arguments to control the
        optimisation algorithm go in the `fit_kwargs` dictionary, fit function key-word arguments go in the `fn_kwargs`
        and given key-word arguments control the `xarray.apply_ufunc` function.

        :param fitter: Fitting object which controls the fitting
        :type fitter: EasyScience.fitting.Fitter
        :param args: Arguments to go to the fit function
        :type args: Any
        :param dask: Dask control string. See `xarray.apply_ufunc` documentation
        :type dask: str
        :param fit_kwargs: Dictionary of key-word arguments to be supplied to the Fitting control
        :type fit_kwargs: dict
        :param fn_kwargs: Dictionary of key-words to be supplied to the fit function
        :type fn_kwargs: dict
        :param vectorize: Should the fit function be given dependents in a single object or split
        :type vectorize: bool
        :param kwargs: Key-word arguments for `xarray.apply_ufunc`. See `xarray.apply_ufunc` documentation
        :type kwargs: Any
        :return: Results of the fit
        :rtype: FitResults
        """

        # Deal with any kwargs which has been given
        if fn_kwargs is None:
            fn_kwargs = {}
        if fit_kwargs is None:
            fit_kwargs = {}
        old_fit_func = fitter.fit_function

        # Wrap and broadcast
        bdims, f = self.fit_prep(fitter.fit_function)
        dims = self._obj.dims

        # Find which coords we need
        if isinstance(dims, dict):
            dims = list(dims.keys())

        # Wrap the wrap in a callable
        def local_fit_func(x, *args, **kwargs):
            """
            Function which will be called by the fitter. This will deal with sending the function the correct data.
            """
            kwargs['vectorize'] = vectorize
            res = xr.apply_ufunc(f, *bdims, *args, dask=dask, kwargs=fn_kwargs, **kwargs)
            if dask != 'forbidden':
                res.compute()
            return res.stack(all_x=dims)

        # Set the new callable to the fitter and initialize
        fitter.initialize(fitter.fit_object, local_fit_func)
        # Make EasyScience.fitting.Fitter compatible `x`
        x_for_fit = xr.concat(bdims, dim='fit_dim')
        x_for_fit = x_for_fit.stack(all_x=[d.name for d in bdims])
        try:
            # Deal with any sigmas if supplied
            if fit_kwargs.get('weights', None) is not None:
                fit_kwargs['weights'] = xr.DataArray(
                    np.array(fit_kwargs['weights']),
                    dims=['all_x'],
                    coords={'all_x': x_for_fit.all_x},
                )
            # Try to perform a fit
            f_res = fitter.fit(x_for_fit, self._obj.stack(all_x=dims), **fit_kwargs)
            f_res = check_sanity_single(f_res)
        finally:
            # Reset the fit function on the fitter to the old fit function.
            fitter.fit_function = old_fit_func
        return f_res


def check_sanity_single(fit_results: FitResults) -> FitResults:
    """
    Convert the FitResults from a fitter compatible state to a recognizable DataArray state.

    :param fit_results: Results of a fit to be modified
    :type fit_results: FitResults
    :return: Modified fit results
    :rtype: FitResults
    """
    items = ['y_obs', 'y_calc', 'residual']

    for item in items:
        array = getattr(fit_results, item)
        if isinstance(array, xr.DataArray):
            array = array.unstack()
            array.name = item
            setattr(fit_results, item, array)

    x_array = fit_results.x
    if isinstance(x_array, xr.DataArray):
        fit_results.x.name = 'axes_broadcast'
        x_array = x_array.unstack()
        x_dataset = xr.Dataset()
        dims = [dims for dims in x_array.dims if dims != 'fit_dim']
        for idx, dim in enumerate(dims):
            x_dataset[dim + '_broadcast'] = x_array[idx]
            x_dataset[dim + '_broadcast'].name = dim + '_broadcast'
        fit_results.x_matrices = x_dataset
    else:
        fit_results.x_matrices = x_array
    return fit_results


def check_sanity_multiple(fit_results: FitResults, originals: List[xr.DataArray]) -> List[FitResults]:
    """
    Convert the multifit FitResults from a fitter compatible state to a list of recognizable DataArray states.

    :param fit_results: Results of a fit to be modified
    :type fit_results: FitResults
    :param originals: List of DataArrays which were fitted against, so we can resize and re-chunk the results
    :type originals: List[xr.DataArray]
    :return: Modified fit results
    :rtype: List[FitResults]
    """

    return_results = []
    offset = 0
    for item in originals:
        current_results = fit_results.__class__()
        # Fill out the basic stuff....
        current_results.engine_result = fit_results.engine_result
        current_results.minimizer_engine = fit_results.minimizer_engine
        current_results.success = fit_results.success
        current_results.p = fit_results.p
        current_results.p0 = fit_results.p0
        # now the tricky stuff
        current_results.x = item.EasyScience.generate_points()
        current_results.y_obs = item.copy()
        current_results.y_obs.name = f'{item.name}_obs'
        current_results.y_calc = xr.DataArray(
            fit_results.y_calc[offset : offset + item.size].data,
            dims=item.dims,
            coords=item.coords,
            name=f'{item.name}_calc',
        )
        offset += item.size
        current_results.residual = current_results.y_calc - current_results.y_obs
        current_results.residual.name = f'{item.name}_residual'
        return_results.append(current_results)
    return return_results
