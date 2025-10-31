from enum import Enum, unique

FACTORY_SUPPORTED_NET_MAP = {
    "mlp": "LinearBlock",
    "linear_block": "LinearBlock",
    "linearblock": "LinearBlock",
    "lstm_mlp": "LstmMlp",
    "lstmmlp": "LstmMlp",
    "gru_mlp": "GruMlp",
    "grumlp": "GruMlp",
    "recurrent_mlp": "RecurrentMlp",
    "recurrentmlp": "RecurrentMlp",
    "lstm": "LSTM",
    "gru": "GRU",
    "siren_block": "SirenBlock",
    "gaussian_rbf": "GaussianRbf",
}

# Classification according to parameter structure
LINEAR_BLOCK = {"LinearBlock", "SirenBlock", "CapsLinear"}
RECURRENT_MLP = {"LstmMlp", "GruMlp", "RecurrentMlp"}
RECURRENT = {"LSTM", "GRU"}
LSTM_NET = {"LSTM", "LstmMlp"}
GRU_NET = {"GRU", "GruMlp"}
RBF_NET = {"Rfb", "GaussianRbf"}
CONV_NET = {}


@unique
class NetTypeFLag(Enum):
    LinearBlock = 0
    RecurrentMlp = 1
    Recurrent = 2
    RbfModel = 3


def get_net_type(name: str) -> NetTypeFLag:
    if name in LINEAR_BLOCK:
        return NetTypeFLag.LinearBlock
    elif name in RECURRENT_MLP:
        return NetTypeFLag.RecurrentMlp
    elif name in RECURRENT:
        return NetTypeFLag.Recurrent
    elif name in RBF_NET:
        return NetTypeFLag.RbfModel
