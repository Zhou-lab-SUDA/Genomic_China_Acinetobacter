"""
Scenarios Module

Available scenarios:
- invasion: Clade 2.5 invading Clade 2.4
- covid_intervention: COVID-19 three-phase differentiated intervention
# - sensitivity: Sensitivity analysis (to be implemented)

Usage:
    # Method 1: Run the script directly
    python scenarios/invasion.py --introduction-time 300
    python scenarios/covid_intervention.py --intervention-start 300 --recovery-start 800

    # Method 2: Import in code
    from scenarios.invasion import run_invasion_scenario
    from scenarios.covid_intervention import run_covid_intervention_scenario
    results = run_invasion_scenario(...)
    results = run_covid_intervention_scenario(...)
"""

__all__ = [
    'invasion',
    'covid_intervention',
]
