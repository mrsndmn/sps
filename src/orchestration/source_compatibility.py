"""Preserve source identities across reviewed relocations."""

from hashlib import sha256
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFINITIONS = _REPO / "src/experiments"
_SOURCE_DEPENDENCIES = {
    "src/workloads/evaluation/teacher_cap_probe.py": (
        "instruct_config.py",
        "vllm_sps_config.py",
    ),
    "src/workloads/evaluation/qwen35_cap_compare.py": (
        "instruct_config.py",
        "teacher_cap_probe.py",
    ),
}
# Exact digests cover the relocated implementations and their typed settings.
_MIGRATED_SOURCES = {
    (
        "src/data/instruct_data.py",
        "6ec6a8dceb17c652f8239bc4163ae7939931a14122293e99d5774bd5596d7a75",
    ): "f0958b9beca04a59f079325d4c8ad1da0e9467f38383e52edd21820eac04c3a7",
    (
        "src/experiments/vllm_sps_config.py",
        "2aceb42cc4b3d1831588d41d58b6ab5d1fc681e20dc439f4016799ddf562206b",
    ): "34e526a7968fd7cbf9fa8e0f0da3a5f044fc8aafbc6bc2e14abad30a2172da76",
    (
        "src/workloads/adaptation/base50_config.py",
        "6303a232a0abcff7372c160a5595b782234d8a18ecea4230768d45a4e7d0c1c5",
    ): "ca1dc73f50ea1332bc49c0fb39d2d1d351e1f78511db2c5958f4d0a59be04d5a",
    (
        "src/workloads/adaptation/base50_run.py",
        "a7d84108e8430ae29dd24b18926ab4a68efe6e98cfe7c410eca6cccc35c54304",
    ): "8a926e14d3d584cc2126d0e35fd17ff667435bbddf290856c35633dba5e79250",
    (
        "src/workloads/adaptation/base50_train.py",
        "da6c433b56f9f6dd9408d7cb6f5283986618e3dbcdc8e1c05b5a53768fa7e04b",
    ): "eb79f69eabfa8f36a7e22536fd3aa86cced8770a7de031edaaac6a7ea723d977",
    (
        "src/workloads/distillation/dflash_mask_ablation.py",
        "34274ee8a8257409be21441403980f6192978337b3e12a4d5b3ddcad164b0896",
    ): "83d9e1f0448858b963db5cc0801f641f4f85d21d09b3987762c74e09e287ec5b",
    (
        "src/workloads/distillation/dflash_repro_eval.py",
        "a761d6b20cbc4e5fea7b74a80b0074f6c83e4d07a224fb680a3ee99344292cdd",
    ): "17fb81dc97b11a35941aeafb175a00e5e6130931bd71caec9122bb82ed2b83fa",
    (
        "src/workloads/distillation/dflash_repro_train.py",
        "7f0b28b8c19862e13883c38966704a6c6cad7c2fb0ac12ffbce0fa62ed55c554",
    ): "3d838ef07c5e0a345999059bd9181571d58a873c6107b1eb7ecffc5606b244b0",
    (
        "src/workloads/distillation/qwen4b_drafter_data.py",
        "214f51feabd0eb795c2c8b02d5c553e430d6cf56cf6ba88ae0871190217e8281",
    ): "abbed60acc831f5bc11f08bb42b060ea3fbec81950ef55e84864e192c9fe5960",
    (
        "src/workloads/distillation/qwen4b_drafter_eval.py",
        "ea8d55f3889dcbbc7d7eac984cbfb22940d84d1667f5a8162c51ebc8121f3a07",
    ): "1534d6b9aca6251a9415229be467bd3e37fa652ac6dcf516e0e32a44ea36d21b",
    (
        "src/workloads/distillation/qwen4b_drafter_train.py",
        "403ebcc0d52097ee1a246414a9bd5268c01315256f3a941017c02e3ce0ec3fde",
    ): "d11762dcd5668935bfca494ea348ed73200fc70372894b94e7a1c77d9f788dc2",
    (
        "src/workloads/evaluation/qwen35_cap_compare.py",
        "89a9ddfb3dccf0fae363d4be5155a678a3781db67b7588346e325a53562c38ab",
    ): "368b7c39020d6f4dda4a336ec92f8dc221e24ca187bf6539d9aec3dea2380e12",
    (
        "src/workloads/evaluation/report_mmlu_stem.py",
        "e4430b6830ea58914041e28891ad6def29867e7a3e7cbe89fe08aa5e546fc6c8",
    ): "087b63226664a1efef8a233814f89a8a3de1e7f44245da8924263f8adf16a6b6",
    (
        "src/workloads/evaluation/teacher_cap_probe.py",
        "692b2c4e427634949cfc62bd82d197a037b679f5d665b6a215b8b663f7e6230b",
    ): "5bb50eb67bff7ec297e740ea8bffdce6d220f9bf5051cc47d67a5755634f82c5",
    (
        "src/workloads/instruction/extend_common.py",
        "169a3dce583d88ec3c7f59e188a03d8dc41753f2562b440cbde319e82c39e604",
    ): "fe8708641d50e03ebb986e46b6a34d0a25a3275299ea53b05a1c95778e4132b5",
    (
        "src/workloads/instruction/extend_eval.py",
        "31cc64a3bd693a848e96f7306a0e7419ee66cc6d6c4a6ad9839a69d530d2233e",
    ): "108e10d303a62f2bbdf7f84ab550372fb5d0acaa7c239e3febcdddcf48014447",
    (
        "src/workloads/instruction/extend_run.py",
        "24c7870cb8c77ac8c3faa9c5c9b03e367864354dc8c00e2df9da2f81137922e6",
    ): "30eeb53663728010dc04f78d1f2685029913e55540c8780fa441668e61324cd6",
    (
        "src/workloads/instruction/extend_train.py",
        "a448f6e2d7cb58f3b8a49808ab1de5c357b764f55ca55c1929da8b0bdc291869",
    ): "9d7aa54342df7401cc8bd7421507bdc6e1a751d84e77c7e45d29b9c3b795e2fd",
    (
        "src/workloads/instruction/ifeval_run.py",
        "9a6c370d148fe9496dccc6d226faf574b650bcdf618a8c0048801c8e0599d777",
    ): "ef2d88a52b685bbf15844ee2e6d7c189afff5044d8c6e9bb2526901e9d9904bb",
    (
        "src/workloads/instruction/instruct_common.py",
        "c7be760b5194e1d239d7bfd0eca59fa93e72da714ef8d8e419791939df2d431c",
    ): "e51cde4cacb8d0d9523731fcc36cbec9ff2acbde7ebb4bf5ad74344ebc6e2ab9",
    (
        "src/workloads/instruction/instruct_evaluate.py",
        "b7077e80ce43556fc60b3baab2519c18892ee563145277910c789512a7ceae47",
    ): "be75218314b57bedaeaedd62a1cb87c9c6762f6aa8b7ef2cdaf425111b3b6897",
    (
        "src/workloads/instruction/instruct_generation.py",
        "459430c1eb805aa88f955fb35d7bfbda287201da64f56e0a3c5e082ec2b5785f",
    ): "3e654ca596ff3a719b8d91bc1f2582b1e3cfa402947378e81b09c2a858048b10",
    (
        "src/workloads/instruction/instruct_train.py",
        "98aa40a25744f65bc8d729bc0816a8e4c111ed0d44c3a5a619cbf2004990e752",
    ): "9500513fb7b75139ea36cb0afd456289f57290ccf32b3cfd8ae5ee82ef53180e",
    (
        "src/workloads/instruction/rerun_eval.py",
        "36d2ec574c56d1148402b98a518e9910b067890aa8f1ea5201f625d3d0d4b489",
    ): "32d2267edffc3fe0c55c9eaa15d27ff94def726e1f235cb9ae2180075a062800",
    (
        "src/workloads/instruction/rerun_jobs.py",
        "981cc85b75962458b48d33fc9c5fdc8d3c5078cce44afe5d7bbb2b5b3f025c56",
    ): "3404403be8cfc1f04551bacd4ba42a60d21839000653a92d1981f2330ee4b6f5",
    (
        "src/workloads/instruction/sched_common.py",
        "b31a1bde5cddb365a78d97d2b437891b5660cf7914b6e56a63f582d3a3b87146",
    ): "279065fcb9c270eadbe781aba2d0b6dcd7e28aff3d1d7bfced933793640c3af8",
    (
        "src/workloads/instruction/sched_eval.py",
        "51fbb70d979f45a779c52b53545165b7de0c7482e4f8d63c1239b886df7cf8bd",
    ): "2f6a40683f400c57ca162716fcf261d4231863aeb953e0354be7a84ef0e3c84c",
    (
        "src/workloads/instruction/sched_run.py",
        "760bf2757aaf0c6eb6b4cd1f6c2eb206a19567eecdc90be23d8e5a48418de177",
    ): "9624e8193c76805ba39b35da2ae6dd9d24e93a426708fefc5fc2aae0e45433a4",
    (
        "src/workloads/instruction/sched_train.py",
        "bd3ab16f20130ea54848d5d88495bb8a4bcff7b2fafcb2b8e6937004dcd17037",
    ): "6f9eeab138019cad2be126a125b32b763dc44bc8ac9d7e5e133a8a75f45ed2f2",
    (
        "src/workloads/provenance/provenance_evaluate.py",
        "e13cf7d0ed8841de5fcc888273902a5b61d8bb563b7361cca36930fad6896a22",
    ): "e90d039f6d46e02e87abd91fcaea5cd975534ac3ba4797ef73c297c5345e34a8",
    (
        "src/workloads/provenance/provenance_prepare.py",
        "40fd99fa01fa34c9ea8256c942231eb7ea725938a52e84894714205680fa5ea9",
    ): "1ee26d8274b6171cbdfc14a973d33c90d95f7ade888392831badca58127a8a27",
    (
        "src/workloads/provenance/provenance_train.py",
        "7d2b219010e258eb66dbfd618ffef19ddaf673938009497d5d4edc5aee9f7a63",
    ): "1f01b9efcf490ee730781edcccdc21f385560b8018c82e1554d14103a51ae938",
    (
        "src/workloads/readout/real_readout_eval.py",
        "0bcbdef8c525f313e9253d8847817e6de1949b68c344339f1c2bc40e5a0eae60",
    ): "1e90ba6331dbff28c930f3eb32646ed9ea319256b11083fb5b795f467768864b",
    (
        "src/workloads/readout/real_readout_gsm8k.py",
        "5bc2230b30e493b194a8af1c8e0c696ed37cf69030efd2e0d63f921b11a212ff",
    ): "2fd3b9337145a69afc29c788c2b1d3e86ac0455a215becc3a34a3196c0db44d3",
    (
        "src/workloads/readout/real_readout_humaneval.py",
        "730a0582aae16571da5cde74ba9424013fc073018dbd121aa221fb1693e77d16",
    ): "8284d2848419985bdc2830349cef7c5361c772fd5d95b1c1a695e2418643de53",
    (
        "src/workloads/readout/real_readout_mmlu.py",
        "3f65a800fb8670e2acabbcdb950ef0df508146c74c091f6b7d3ee7f1eb2cb422",
    ): "e686a7040dfb97e0cb3b74cf68d7d4c3dee653b6051716887550cdaca22b5759",
    (
        "src/workloads/readout/real_readout_model.py",
        "56beb7bee4b0651a1cde8ec23968fd2c30316f33e30f746d74d3ed6285b85a22",
    ): "42a90e05db4cf82920982db2ccef2ab71f12df997373fb7e12d1f45517391f65",
    (
        "src/workloads/readout/real_readout_serving.py",
        "80252830ac201afab1cf48d71f1db9f3a41aac84fe52a244ff0e1bcf89321154",
    ): "c4d65a4f0c7006b627ee6a94a00a85531fd6654ae172d056158854f194a6e367",
    (
        "src/workloads/serving/draft_sweep.py",
        "69eb0c3f06a0c2d674e89aafd7ea6e1cd48e1db1a656d16c09608aa72951e606",
    ): "cbd47a8f72edc0eef7b7c76d0e3183771bdf59332ca5471c4b5aeab9e01bba6e",
    (
        "src/workloads/serving/sampling_audit.py",
        "d84586b8b86432fe6741d960b44e4ee923f4ee9731af297bb7f1622ab52ba1b7",
    ): "79fad8be3df0f5234baeba5036d315584c2bcb6c89b98b301b9f1034d57e1b75",
    (
        "src/workloads/serving/sampling_backbone_precision.py",
        "5403a37d43b592e0ec89f7f9a6581a1f65e946ed2f6dbbc0d208897f6843faab",
    ): "fc016300bdb5bb69a6ed22999931377e8ccb7ec5d054a4ba4b765bddb9247d17",
    (
        "src/workloads/serving/sampling_cache_trace.py",
        "360fcdfbe4b6c52d920cdc034677d96b3e35389477a32c86876f90f4c9a4244a",
    ): "f860abeca174b22e33e343ea67246c4790da7f4bfe48af39449cdfd3feab56ef",
    (
        "src/workloads/serving/sampling_precision.py",
        "b6aa6cd392a1b8e28663215f297cf69d9037df03523cb667d566bc0f214a9731",
    ): "ade8f569031522bd0e1e9a33644f132ec3ec7adafb8c11b8d0a766dbb3b92681",
    (
        "src/workloads/serving/sampling_trace.py",
        "f96b1d0b38f50fe61a04692f85e9e556999ec296b960bff1af6769f0ebad6d70",
    ): "698b8f7ececf4775f67b746c8d4ddb30c01a7be3d3e3984f7ea9788cb53593d6",
    (
        "src/workloads/serving/serving_benchmark.py",
        "dadc5988275f91f968544ddffbdb0185eb87d1185e678d946368fc45c8e07fdf",
    ): "29217e8596d658ec23dc0c9b884568b0e3fa863e71aff3ba17bd71ba7b5620b0",
    (
        "src/workloads/serving/serving_graphs.py",
        "3d5d1fc06aaee40f328c11f25e38b20ba52a36b4714f3ea43e69c17ff5640f41",
    ): "f25cf85077c5d38a596f884d5d1b89939140c95e2a57af6a2f4adf5f8f254c5b",
    (
        "src/workloads/serving/serving_runtime.py",
        "35f4ec2f190af46a3914d36ec7806999f8a54ab7e1f75e0c5e9d9b6ef8f26b1b",
    ): "44bb56bfca35b7b69eaba91219387c4dae628d180687707dc6a8c78ff774dd59",
    (
        "src/workloads/serving/vllm_sps_audit.py",
        "8c5ffb053f42301e6ba27fa1cb6c925ed48a919da11dd893bf46f72fbaab5f68",
    ): "3a119a43ed6d21988f54de9afda56421278e3bfe251fa43c736cd82ecab78a23",
    (
        "src/workloads/serving/vllm_sps_common.py",
        "1ea46c8d45b27b1622f1206bb3b4e8e297b732ca73d57a3a115871304706c2e8",
    ): "92a61c5518618da7339139191e68d2225a47fa48e9c595827e6c47f5f3b45e17",
    (
        "src/workloads/serving/vllm_sps_evaluate.py",
        "aa0946737c3666746bea43731656a424e810c849d5330977be0fed6bcc1f2ac8",
    ): "7f30a8e8f2e3efcf8f3c951d6f19c8e0dceccccf60a47eb1a7500d4e95fde96c",
}


def source_hash(path):
    """Use the recorded identity only for an exact, reviewed source migration."""
    path = Path(path).resolve()
    digest = sha256(path.read_bytes()).hexdigest()
    if not path.is_relative_to(_REPO):
        return digest
    relative = str(path.relative_to(_REPO))
    if relative.startswith(("src/workloads/", "src/data/")):
        name = "base50.py" if path.name == "base50_config.py" else path.name
        definition = _DEFINITIONS / name
        if definition.is_file():
            digest = sha256(
                bytes.fromhex(digest) + sha256(definition.read_bytes()).digest()
            ).hexdigest()
    for dependency in _SOURCE_DEPENDENCIES.get(relative, ()):
        digest = sha256(
            bytes.fromhex(digest)
            + sha256((_DEFINITIONS / dependency).read_bytes()).digest()
        ).hexdigest()
    return _MIGRATED_SOURCES.get((relative, digest), digest)
