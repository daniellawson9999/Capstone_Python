from enum import Enum,auto

#Enum that contains types of networks
#SA_TO_Q = a network with state-action input that ouputs a single q value
#S_TO_QA = a network with state input that ouputs a q value for each action
#SM_TO_QA = a network with multiple state input (frame stacking) that outputs a q value for each action
class Network(Enum):
    SA_TO_Q = auto()
    S_TO_QA = auto()
    SM_TO_QA = auto()


class Networks(Enum):
    DOOM_CNN_SM = auto()
    DUELING_SM = auto()
    DUELING_S = auto()
    DUELING_LSTM_SM = auto()