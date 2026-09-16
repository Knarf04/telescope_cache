"""
Building blocks for telescoping attention, one module per concern.

    range_spec.py    the frozen contract and its branch-free closed forms
    position.py      post-summary RoPE + the learned summary-bin bias
    summaries.py     merge weights, short causal conv, dyadic tree builder
    packing.py       PackedKV and the level-major packing
    sink_gate.py     learned attention sink, output gate
    decode_state.py  per-level rings, capacities, incremental tree update
    flex_common.py   plumbing shared by flex_tree's prefill and decode paths

`reference.py` holds the algorithms built from these and re-exports every
name. This package imports nothing from outside itself.

minimal_reference.py imports range_spec and NOTHING ELSE, on purpose: it is
an INDEPENDENT oracle, and mini<->ref only has force while the two share the
contract and no implementation.
"""
