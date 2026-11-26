"""Source identities for files whose definitions moved into src/experiments."""

from hashlib import sha256
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DEFINITIONS = _REPO / "src/experiments"
_MIGRATED_SOURCES = {
    (
        "scripts/instruction/instruct_train.py",
        "f382159044bb04231115fb1194463f2649bce72efa3f14550c6cf8dd5e2dfb97",
    ): "9500513fb7b75139ea36cb0afd456289f57290ccf32b3cfd8ae5ee82ef53180e",
    (
        "scripts/adaptation/base50_config.py",
        "6b3d2e45f9c7cd0206f9cb14aa4afa6e3262d6c2f956f66a959f02f796872e18",
    ): "ca1dc73f50ea1332bc49c0fb39d2d1d351e1f78511db2c5958f4d0a59be04d5a",
    (
        "scripts/distillation/dflash_mask_ablation.py",
        "f4f7f8c47c9e04fcc09ae572a8179d22aaa17ec174651c9bfcdca914a1ccf67f",
    ): "83d9e1f0448858b963db5cc0801f641f4f85d21d09b3987762c74e09e287ec5b",
    (
        "scripts/distillation/dflash_repro_eval.py",
        "52affc9cccea486886a0d37cc7ebf7426ec554fe2239fde73ac0fcaf64357de7",
    ): "17fb81dc97b11a35941aeafb175a00e5e6130931bd71caec9122bb82ed2b83fa",
    (
        "scripts/distillation/dflash_repro_train.py",
        "21191f2075f06da2c86ffa7431ec868f7f4d55fe80a865e2341b27817cc0b6ff",
    ): "3d838ef07c5e0a345999059bd9181571d58a873c6107b1eb7ecffc5606b244b0",
    (
        "scripts/distillation/qwen4b_drafter_data.py",
        "7998895ac54c9e631fb08b072000ac6f177291e3ddbe33f711db394bba4cbc0c",
    ): "abbed60acc831f5bc11f08bb42b060ea3fbec81950ef55e84864e192c9fe5960",
    (
        "scripts/distillation/qwen4b_drafter_eval.py",
        "fb95b93688e281c75ea5ec2d7600a499c1d3287b262108cb6879501ab5723741",
    ): "1534d6b9aca6251a9415229be467bd3e37fa652ac6dcf516e0e32a44ea36d21b",
    (
        "scripts/distillation/qwen4b_drafter_train.py",
        "882bf0c37a1ff28f65c04e8e6493e10cc71e97509446f9a3de54b0c7b439512e",
    ): "d11762dcd5668935bfca494ea348ed73200fc70372894b94e7a1c77d9f788dc2",
    (
        "scripts/evaluation/qwen35_cap_compare.py",
        "599e3deb53ea3111183d2e2e658df26953377caab2c73b4e7998c568f51e266a",
    ): "368b7c39020d6f4dda4a336ec92f8dc221e24ca187bf6539d9aec3dea2380e12",
    # Shard support in c84d1ea left the original full-cohort report design
    # unchanged. Use the published report identity for this exact migration.
    (
        "scripts/evaluation/teacher_cap_probe.py",
        "3371a9b42d15516186739965fa7be24872be2e93b7f3eee759ab17e2dbdcd50c",
    ): "5bb50eb67bff7ec297e740ea8bffdce6d220f9bf5051cc47d67a5755634f82c5",
    (
        "scripts/instruct_common.py",
        "9e9632a7380e3655dd2dccecfc0422f29a0711977f4e3340d5e015ade67f2abb",
    ): "e51cde4cacb8d0d9523731fcc36cbec9ff2acbde7ebb4bf5ad74344ebc6e2ab9",
    (
        "scripts/instruction/extend_common.py",
        "c7a23281fd9a46870649d15743ebeb520bb1b3137cd87d3d01a803f34fad8694",
    ): "fe8708641d50e03ebb986e46b6a34d0a25a3275299ea53b05a1c95778e4132b5",
    (
        "scripts/instruction/ifeval_run.py",
        "eede8a221c939a4f52edf4744698e30b35386faac766065e006dd4e9e2e8d05d",
    ): "ef2d88a52b685bbf15844ee2e6d7c189afff5044d8c6e9bb2526901e9d9904bb",
    (
        "scripts/instruction/instruct_data.py",
        "b136d81b000cd776a7c58032257873e0355b0aad5533133caae204729727b296",
    ): "f0958b9beca04a59f079325d4c8ad1da0e9467f38383e52edd21820eac04c3a7",
    (
        "scripts/instruction/instruct_evaluate.py",
        "200edd366ce03ccc0f8289b6578f6c2dba2e152f91ac9f55007147f1a4fed341",
    ): "be75218314b57bedaeaedd62a1cb87c9c6762f6aa8b7ef2cdaf425111b3b6897",
    (
        "scripts/instruction/sched_common.py",
        "b1b5eac693bef953b0a752815a0b23019cc916a1c16d295cfd08e89b433506cc",
    ): "279065fcb9c270eadbe781aba2d0b6dcd7e28aff3d1d7bfced933793640c3af8",
    (
        "scripts/provenance/provenance_evaluate.py",
        "431028c19a6a04bb2a72409bf33bca706f9bd64000bdb08d9918d060f24be009",
    ): "e90d039f6d46e02e87abd91fcaea5cd975534ac3ba4797ef73c297c5345e34a8",
    (
        "scripts/provenance/provenance_prepare.py",
        "1d35db04de3d274d6689f8c8bd23f18dba876970e6a3b8774b673bcfac3800be",
    ): "1ee26d8274b6171cbdfc14a973d33c90d95f7ade888392831badca58127a8a27",
    (
        "scripts/provenance/provenance_train.py",
        "ac475ecddee206e7cf95e39c4a16054d3105d8f08328f6b87e6fcb09fdda2859",
    ): "1f01b9efcf490ee730781edcccdc21f385560b8018c82e1554d14103a51ae938",
    (
        "scripts/readout/real_readout_eval.py",
        "2e2df3cc17569de893eb09b922a62026e5a2f2305580a000acb5b27d6d8c1437",
    ): "1e90ba6331dbff28c930f3eb32646ed9ea319256b11083fb5b795f467768864b",
    (
        "scripts/readout/real_readout_gsm8k.py",
        "938992029b58159c37d01248bcbcbc1b8e098d9b6f5b127a44c38625860e1c60",
    ): "2fd3b9337145a69afc29c788c2b1d3e86ac0455a215becc3a34a3196c0db44d3",
    (
        "scripts/readout/real_readout_humaneval.py",
        "aaed763ff13146dca2fdb1c01756d4b675b36eac28eb87e119d0edbf34ec9bb1",
    ): "8284d2848419985bdc2830349cef7c5361c772fd5d95b1c1a695e2418643de53",
    (
        "scripts/readout/real_readout_mmlu.py",
        "c4658315a3fcbe601432389125cb54a523259523c39c09562ed831dc2595eea1",
    ): "e686a7040dfb97e0cb3b74cf68d7d4c3dee653b6051716887550cdaca22b5759",
    (
        "scripts/readout/real_readout_serving.py",
        "1aef693d8b392498d4ae8eba53258a01c69d691b570b36bc27299c023622fa8f",
    ): "c4d65a4f0c7006b627ee6a94a00a85531fd6654ae172d056158854f194a6e367",
    (
        "scripts/report_mmlu_stem.py",
        "3ec8ae20bc662b783bdec06197eb052d3bf5379c5b2bf75d55ee8db48246d921",
    ): "087b63226664a1efef8a233814f89a8a3de1e7f44245da8924263f8adf16a6b6",
    (
        "scripts/serving/draft_sweep.py",
        "25736f28e565b69c961989b8f98f586fb103679969a43df1e0ebea3d7e9ad228",
    ): "cbd47a8f72edc0eef7b7c76d0e3183771bdf59332ca5471c4b5aeab9e01bba6e",
    (
        "scripts/serving/sampling_audit.py",
        "378d950ebe0d38274282e98d52c287ed16ddea46cd365cec949ea877ac5b1df7",
    ): "79fad8be3df0f5234baeba5036d315584c2bcb6c89b98b301b9f1034d57e1b75",
    (
        "scripts/serving/sampling_backbone_precision.py",
        "cba2951fa1319436d82b1d63b30f45c089e23336021fb426bac65df61f9d1cf4",
    ): "fc016300bdb5bb69a6ed22999931377e8ccb7ec5d054a4ba4b765bddb9247d17",
    (
        "scripts/serving/sampling_cache_trace.py",
        "6795731a693e29873019be989d752e64dd9f53cfa87b4a8efd43668104fcff84",
    ): "f860abeca174b22e33e343ea67246c4790da7f4bfe48af39449cdfd3feab56ef",
    (
        "scripts/serving/sampling_precision.py",
        "15439d428c0fb2986ec15738d1f0577ad331f56e64c091e0f8287146f4de58b3",
    ): "ade8f569031522bd0e1e9a33644f132ec3ec7adafb8c11b8d0a766dbb3b92681",
    (
        "scripts/serving/sampling_trace.py",
        "eb103d87521891e74adc9944520e5211f3f0607c5c38632e087f830c493a21e5",
    ): "698b8f7ececf4775f67b746c8d4ddb30c01a7be3d3e3984f7ea9788cb53593d6",
    (
        "scripts/serving/serving_benchmark.py",
        "6180a85f401c59899a3313b168ce83e546715da312dd1bf60a8b453c3c23b03f",
    ): "29217e8596d658ec23dc0c9b884568b0e3fa863e71aff3ba17bd71ba7b5620b0",
    (
        "scripts/serving/serving_graphs.py",
        "f07d093ae56328b689d2f5e652b2f73103db22d21e4ce3b55f1bd0d2042c4d52",
    ): "f25cf85077c5d38a596f884d5d1b89939140c95e2a57af6a2f4adf5f8f254c5b",
    (
        "scripts/serving/serving_runtime.py",
        "a5bb2074e8a45ed05557a0aab2329e90cdaf84188e0caa5745fccc3c33adcc2b",
    ): "44bb56bfca35b7b69eaba91219387c4dae628d180687707dc6a8c78ff774dd59",
    (
        "scripts/serving/vllm_sps_evaluate.py",
        "d511342da36ba47aa9acf3a737d5f5b65b4f10bf662dd2fe77f04f7cade2464c",
    ): "7f30a8e8f2e3efcf8f3c951d6f19c8e0dceccccf60a47eb1a7500d4e95fde96c",
    (
        "src/experiments/vllm_sps_config.py",
        "2aceb42cc4b3d1831588d41d58b6ab5d1fc681e20dc439f4016799ddf562206b",
    ): "34e526a7968fd7cbf9fa8e0f0da3a5f044fc8aafbc6bc2e14abad30a2172da76",
}


def source_hash(path):
    """Use the recorded identity only for an exact, reviewed source migration."""
    path = Path(path).resolve()
    digest = sha256(path.read_bytes()).hexdigest()
    if not path.is_relative_to(_REPO):
        return digest
    relative = str(path.relative_to(_REPO))
    if relative.startswith("scripts/"):
        name = "base50.py" if path.name == "base50_config.py" else path.name
        definition = _DEFINITIONS / name
        if definition.is_file():
            digest = sha256(
                bytes.fromhex(digest) + sha256(definition.read_bytes()).digest()
            ).hexdigest()
    return _MIGRATED_SOURCES.get((relative, digest), digest)
