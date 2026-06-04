from pathlib import Path
import pytest
from owm.codegraph import parse_repo
from owm.topology import seed_graph

FIXTURE = Path(__file__).parent / "fixtures" / "ob_mini"


@pytest.fixture
def seed_cwm():
    """A freshly seeded CausalWorldModel from the mini Online Boutique fixture."""
    return seed_graph(parse_repo(FIXTURE))
