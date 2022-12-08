class Alternative:
    def __init__(self, *models, curr_model=0):
        self.__curr_model = curr_model
        self.models = models

    def fit(self, X, y=None):
        self.get_curr_model().fit(X, y)
        return self

    def predict(self, X):
        return self.get_curr_model().predict(X)

    def transform(self, X):
        res = self.get_curr_model().transform(X)
        return res

    def set_curr_model(self, model_index: int):
        self.__curr_model = model_index

    def get_curr_model(self):
        return self.models[self.__curr_model]

    def set_params(self, **kwargs):
        if 'curr_model' in kwargs:
            self.__curr_model = kwargs['curr_model']
            del kwargs['curr_model']
        self.get_curr_model().set_params(**kwargs)
        return self


# testing
import unittest.mock as mock
mock0 = mock.Mock()
mock1 = mock.Mock()
mock2 = mock.Mock()

alternative = Alternative(mock0, mock1, mock2)

###################
alternative.set_params(curr_model=0)
alternative.fit(1, 2)
mock0.fit.assert_called_with(1, 2)

mock0.transform.return_value = -1
assert alternative.transform(-2) == -1
mock0.transform.assert_called_with(-2)

mock0.predict.return_value = -3
assert alternative.predict(-4) == -3
mock0.predict.assert_called_with(-4)

###################
alternative.set_params(curr_model=1)
alternative.fit(3, 4)
mock1.fit.assert_called_with(3, 4)

mock1.transform.return_value = -5
assert alternative.transform(-6) == -5
mock1.transform.assert_called_with(-6)

mock1.predict.return_value = -7
assert alternative.predict(-8) == -7
mock1.predict.assert_called_with(-8)

###################
alternative.set_params(curr_model=2)
alternative.fit(5, 6)
mock2.fit.assert_called_with(5, 6)

mock2.transform.return_value = -9
assert alternative.transform(-10) == -9
mock2.transform.assert_called_with(-10)

mock2.predict.return_value = -11
assert alternative.predict(-12) == -11
mock2.predict.assert_called_with(-12)
print("done")