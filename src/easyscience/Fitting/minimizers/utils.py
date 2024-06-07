import numpy as np


class FitResults:
    """
    At the moment this is just a dummy way of unifying the returned fit parameters.
    """

    __slots__ = [
        'success',
        'fitting_engine',
        'fit_args',
        'p',
        'p0',
        'x',
        'x_matrices',
        'y_obs',
        'y_calc',
        'y_err',
        'engine_result',
        'total_results',
    ]

    def __init__(self):
        self.success = False
        self.fitting_engine = None
        self.fit_args = {}
        self.p = {}
        self.p0 = {}
        self.x = np.ndarray([])
        self.x_matrices = np.ndarray([])
        self.y_obs = np.ndarray([])
        self.y_calc = np.ndarray([])
        self.y_err = np.ndarray([])
        self.engine_result = None
        self.total_results = None

    @property
    def n_pars(self):
        return len(self.p)

    @property
    def residual(self):
        return self.y_obs - self.y_calc

    @property
    def chi2(self):
        return ((self.residual / self.y_err) ** 2).sum()

    @property
    def reduced_chi(self):
        return self.chi2 / (len(self.x) - self.n_pars)


class NameConverter:
    def __init__(self):
        from easyscience import borg

        self._borg = borg

    def get_name_from_key(self, item_key: int) -> str:
        return getattr(self._borg.map.get_item_by_key(item_key), 'name', '')

    def get_item_from_key(self, item_key: int) -> object:
        return self._borg.map.get_item_by_key(item_key)

    def get_key(self, item: object) -> int:
        return self._borg.map.convert_id_to_key(item)


class FitError(Exception):
    def __init__(self, e: Exception = None):
        self.e = e

    def __str__(self) -> str:
        s = ''
        if self.e is not None:
            s = f'{self.e}\n'
        return s + 'Something has gone wrong with the fit'
