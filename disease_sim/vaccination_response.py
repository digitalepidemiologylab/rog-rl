from enum import Enum

# Susceptible, Exposed, Infectious, Symptomatic, Recovered/Dead
class VaccinationResponse(Enum):
    VACCINATION_SUCCESS = 0
    CELL_EMPTY = 1
    AGENT_EXPOSED = 2
    AGENT_INFECTIOUS = 3
    AGENT_SYMPTOMATIC = 4
    AGENT_RECOVERED = 5
    AGENT_VACCINATED = 6
