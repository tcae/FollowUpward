
class binance():

    def __init__(self, param_dict):
        has = {
            'fetchDepositAddress': True,
            'CORS': False,
            'fetchBidsAsks': True,
            'fetchTickers': True,
            'fetchOHLCV': True,
            'fetchMyTrades': True,
            'fetchOrder': True,
            'fetchOrders': True,
            'fetchOpenOrders': True,
            'fetchClosedOrders': True,
            'withdraw': True,
            'fetchFundingFees': True,
            'fetchDeposits': True,
            'fetchWithdrawals': True,
            'fetchTransactions': False,
        }

__all__ = [
    'BaseError',
    'ExchangeError',
    'NotSupported',
    'AuthenticationError',
    'PermissionDenied',
    'AccountSuspended',
    'InsufficientFunds',
    'InvalidOrder',
    'OrderNotFound',
    'OrderNotCached',
    'DuplicateOrderId',
    'NetworkError',
    'DDoSProtection',
    'RequestTimeout',
    'ExchangeNotAvailable',
    'InvalidNonce',
    'InvalidAddress',
    'AddressPending',
    'ArgumentsRequired',
    'BadRequest',
    'BadResponse',
    'NullResponse',
    'OrderNotFillable',
    'OrderImmediatelyFillable',
]

# -----------------------------------------------------------------------------


class BaseError(Exception):
    """Base class for all exceptions"""
    pass


class ExchangeError(BaseError):
    """"Raised when an exchange server replies with an error in JSON"""
    pass


class NotSupported(ExchangeError):
    """Raised if the endpoint is not offered/not yet supported by the exchange API"""
    pass


class ArgumentsRequired(ExchangeError):
    """A generic exception raised by unified methods when required arguments are missing."""
    pass


class BadRequest(ExchangeError):
    """A generic exception raised by the exchange if all or some of required parameters are invalid or missing in URL query or in request body"""
    pass


class BadResponse(ExchangeError):
    """Raised if the endpoint returns a bad response from the exchange API"""
    pass


class NullResponse(BadResponse):
    """Raised if the endpoint returns a null response from the exchange API"""
    pass


class AuthenticationError(ExchangeError):
    """Raised when API credentials are required but missing or wrong"""
    pass


class PermissionDenied(AuthenticationError):
    """Raised when API credentials are required but missing or wrong"""
    pass


class AccountSuspended(AuthenticationError):
    """Raised when user account has been suspended or deactivated by the exchange"""
    pass


class InsufficientFunds(ExchangeError):
    """Raised when you don't have enough currency on your account balance to place an order"""
    pass


class InvalidOrder(ExchangeError):
    """"Base class for all exceptions related to the unified order API"""
    pass


class InvalidAddress(ExchangeError):
    """Raised on invalid funding address"""
    pass


class AddressPending(InvalidAddress):
    """Raised when the address requested is pending (not ready yet, retry again later)"""
    pass


class OrderNotFound(InvalidOrder):
    """Raised when you are trying to fetch or cancel a non-existent order"""
    pass


class OrderNotCached(InvalidOrder):
    """Raised when the order is not found in local cache (where applicable)"""
    pass


class DuplicateOrderId(InvalidOrder):
    """Raised when the order id set by client is not unique"""
    pass


class CancelPending(InvalidOrder):
    """Raised when an order that is already pending cancel is being canceled again"""
    pass


class OrderNotFillable(InvalidOrder):
    """Raised when an order placed as a market order or a taker order is not fillable upon request"""
    pass


class OrderImmediatelyFillable(InvalidOrder):
    """Raised when an order placed as maker order is fillable immediately as a taker upon request"""
    pass


class NetworkError(BaseError):
    """Base class for all errors related to networking"""
    pass


class DDoSProtection(NetworkError):
    """Raised whenever DDoS protection restrictions are enforced per user or region/location"""
    pass


class RequestTimeout(NetworkError):
    """Raised when the exchange fails to reply in .timeout time"""
    pass


class ExchangeNotAvailable(NetworkError):
    """Raised if a reply from an exchange contains keywords related to maintenance or downtime"""
    pass


class InvalidNonce(NetworkError):
    """Raised in case of a wrong or conflicting nonce number in private requests"""
    pass
