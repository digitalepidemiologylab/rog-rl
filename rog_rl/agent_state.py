from enum import Enum


class AgentState(Enum):
    # Susceptible, Exposed, Infectious, Symptomatic, Recovered/Dead
    SUSCEPTIBLE = 0  # *
    INFECTIOUS = 1  # O
    RECOVERED = 2  # R
    VACCINATED = 3  # V
