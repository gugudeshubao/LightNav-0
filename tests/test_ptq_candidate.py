import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "ptq_candidate", Path(__file__).parents[1] / "scripts/lightnav_ptq_candidate.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_quantization_recipe_whitelists_language_without_mutating_stock():
    stock = {"quant_cfg": [
        {"quantizer_name": "*weight_quantizer", "cfg": {"num_bits": (4, 3)}},
        {"quantizer_name": "*input_quantizer", "cfg": {"num_bits": (4, 3)}},
    ], "algorithm": "max"}
    result = MODULE.language_quant_config(stock)
    assert result["quant_cfg"][0] == {"quantizer_name": "*", "enable": False}
    assert all("language_model.layers." in r["quantizer_name"] for r in result["quant_cfg"][1:])
    assert len(stock["quant_cfg"]) == 2
    assert result["algorithm"] == "max"


def test_unknown_quantization_recipe_fails_closed():
    with pytest.raises(ValueError):
        MODULE.language_quant_config({"quant_cfg": []})


def test_rvq_agreement_is_independent_of_grounding():
    action = "<act_l0_38><act_l1_186><act_l2_47>"
    assert MODULE.rvq_tokens("<apos_1297><opos_0>" + action) == [38, 186, 47]
    assert MODULE.rvq_tokens("<apos_700><opos_1>" + action) == [38, 186, 47]


@pytest.mark.parametrize("raw", ["", "<act_l1_2><act_l0_1><act_l2_3>",
                                 "<act_l0_1><act_l1_2><act_l2_3><act_l2_4>"])
def test_invalid_rvq_sequence_fails_closed(raw):
    with pytest.raises(ValueError):
        MODULE.rvq_tokens(raw)
