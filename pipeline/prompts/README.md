PANGEA (NeurIPS 2025) generation prompts, one file per domain and stage.

- medqa_*: the official templates from the supplementary code (third_party/pangea/code/medqa_example/template), with only the placeholder names unified.
- gsm8k_* / finqa_*: transcribed from the appendix (Figures 12, 14, 16, 18, 20, 22); line breaks re-flowed and PDF glyph errors fixed (×, →, ≥).

Placeholders: writer {general} = d_g; projection {block} = τ, {seed} = seed question; annotation {question} = q̃.
Each stage's reply is primed with an assistant prefix (pipeline/pangea_gen.py PREFIX), as in the official code.
