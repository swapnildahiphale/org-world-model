"""The world model as a Predictor. predict() runs do(change, s_t); learn()
applies an incident's outcome. learning_enabled=False is the H2 ablation.
"""
from __future__ import annotations

from owm.groundtruth import Prediction
from owm.propagate import predict_blast
from owm.learn import observe


class OWMEngine:
    def __init__(self, cwm, learning_enabled: bool = True):
        self.cwm = cwm
        self.learning_enabled = learning_enabled

    def predict(self, scenario, pack=None) -> Prediction:
        probs = predict_blast(
            self.cwm,
            touched_services=list(scenario.touched_services) or None,
            touched_configs=list(scenario.touched_configs) or None,
            s_t=scenario.s_t,
        )
        return Prediction(scenario.id, probs)

    def learn(self, scenario, impacted_services, pack=None) -> None:
        """Apply one incident's outcome. Week-1 deepening targets the hidden
        config-class couplings (the surprise/Beta path in learn.observe)."""
        if not scenario.touched_configs:
            return
        origin = scenario.touched_configs[0]
        pre = self.predict(scenario).probs
        observe(self.cwm, origin, impacted_services, pre,
                learning_enabled=self.learning_enabled)
