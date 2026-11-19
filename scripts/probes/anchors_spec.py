"""Writer-stage anchor specs shared by the G1' (AR, span by span) and G3 (Dream canvas) probes.
Each spec: assistant prefix and a list of (anchor text, span length cap in tokens)."""

HEADS = {  # v1: field headers only, 30 tokens per field
    "gsm8k": ("Quantitative Building Blocks:",
              [(f"\n- {h} - ", 30) for h in ["Scenario", "Symbols", "Equations", "Target", "Plan"]]),
    "medqa": ("Clinical Building Blocks:",
              [(f"\n• {h} – ", 30) for h in ["Demographics", "Timeline", "Key clinical events",
                                              "Lab / imaging patterns", "Pathophysiologic principles"]]),
}

RICH = {  # v2: fixed parts of PANGEA's format as anchors, per-field span caps
    "gsm8k": ("Quantitative Building Blocks:", [
        ("\n- Scenario - ", 32),
        ("\n- Symbols - A = ", 40),
        ("\n- Equations - (1) ", 12), (" (2) ", 12), (" (3) ", 12), (" (4) ", 12),
        ("\n- Target - T = ", 16),
        ("\n- Plan - (1) ", 10), (" → (2) ", 10), (" → (3) ", 10), (" → (4) ", 10),
    ]),
}

SPECS = {"heads": HEADS, "rich": RICH}
