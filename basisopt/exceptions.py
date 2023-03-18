# exceptions


class MethodNotAvailable(Exception):
    def __init__(self, estr):
        Exception.__init__(self)
        self.method_str = estr


class PropertyNotAvailable(Exception):
    def __init__(self, pstr):
        Exception.__init__(self)
        self.property_str = pstr


class EmptyCalculation(Exception):
    pass


class FailedCalculation(Exception):
    pass


class ElementNotSet(Exception):
    pass


class EmptyBasis(Exception):
    pass


class InvalidResult(Exception):
    pass


class InvalidMethodString(Exception):
    pass


class DataNotFound(Exception):
    pass


class InvalidDiatomic(Exception):
    pass
