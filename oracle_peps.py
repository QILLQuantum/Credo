class PepsOracle:
    def __init__(self, steps=12, Lz=6, bond=8, use_gpu=False):
        ...

    def run(self, mode="light"):  # light=fast, full=with animation
        # Run our simulation logic
        # Return:
        {
            "certified_energy": -1.2345,
            "uncertainty": 0.0123,
            "logical_advantage": 0.87,
            "code_distance": 12,
            "final_avg_bond": 46.8,
            "growth_rate": 412,
            "lz_boost_effect": "strong (Lz=6)",
            "timestamp": "..."
        }