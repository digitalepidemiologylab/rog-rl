from enum import Enum


class VaccinationResponse(Enum):
    VACCINATION_SUCCESS = 0
    CELL_EMPTY = 1
    AGENT_EXPOSED = 2
    AGENT_INFECTIOUS = 3
    AGENT_SYMPTOMATIC = 4
    AGENT_RECOVERED = 5
    AGENT_VACCINATED = 6
    AGENT_VACCINES_EXHAUSTED = 7
    SIMULATION_NOT_RUNNING = 8
    NO_AGENT = 9
