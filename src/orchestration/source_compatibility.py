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
    # Directory publication and typed family settings.
    ("src/data/pilot_data.py", "de1030ded2dfc2f289e6ab5c5275bbb9f1e7eb554adf75a7e90572fc7d98e08c"): "ec2d159eedfdeacaa71b3a8272a2b8cdec3f73f3a00030ad4470303635ef3a5f",
    ("src/experiments/ar_sps_config.py", "8ca136123935e897cab424b4a0e49adf37ec5127fe4ee885ee9eecb9795bfd34"): "fe27779f0ea01242def1f6145922c64e2606ad2cbb5384fc01a7cd771aaf411a",
    ("src/experiments/base50.py", "60ea72384a222b8ee8f89321cd883f1368254cef447e590ed473dfb02b2d9ef7"): "71ccb29abb3cc92f12a401c4bbd803202932781f87d8ac09f2f5a6e7dacdbb44",
    ("src/experiments/dflash_kd_train.py", "6b957141f2bf44ae1bded7ad11a7cc4debc29aa67dfe016db66f57684360c5fe"): "435c7f6dbd958ac7ce779dd3c19141ea27ecd72810dc71e8a6f1135de85af3af",
    ("src/experiments/dflash_mask_ablation.py", "e645448397369abc7d0b2b842394b329f0d6266942f81b073402f9755d952f5c"): "119ebcd164b2b7daf929c2d0baff43eb91b89e9809a107701a76e9834ac0ac89",
    ("src/experiments/dflash_repro.py", "7cde7698ab462d66539594f9fbea3fc53718d93287d4bc51e946222ca6fd1bd0"): "0369437c2e9b18cc7a3a6eae212e6d77a2a7d2215f29a2f45caf1532b604a9e0",
    ("src/experiments/dflash_repro_eval.py", "74e019131970dd826e0194b318db975b889938f31587d55e1c1b0c81878b8726"): "489d308689c5492dd5ceaa74cdd506b551b6e5e33614d21976b87ac7eaf61711",
    ("src/experiments/dflash_repro_train.py", "562bcfbf72adf89604481608f834927e9400c5388fca1dda1b13f710938c44f6"): "6b92d7f9dde23fcb445fafb917a463c795d505dd9172ba187e4aa0af7da5dddc",
    ("src/experiments/draft_sweep.py", "5d3346150acdf1de723a2ee1529f0f128a622690d2d55bd5408a39e1e798b18c"): "2d1bcf36d855fc55488f5bff29abeb13c17179dfac15601ce7c858907269b05a",
    ("src/experiments/draft_sweep_train.py", "e9bd3d2aa4363b98f6ac51edfed29e08bc067e4792eafbe1cd54aa71121eccb8"): "3fc3d6290bd51ff808b3872e4a80ba17e6a15d15a62d3d035a5a2c9f8428f37f",
    ("src/experiments/extend_config.py", "4494637edb288454dd6844d268fd81da513ec50da0fc0548077d9baf1afc1900"): "e24a18345966e374f96128114d5e186fa7bbaf292d198feab0dee8621b5c9ec8",
    ("src/experiments/ifeval_config.py", "d43d0d04e3684ccc46c442e9e7ca6802fb1a84b58be8633e4628f8285d918605"): "79a3cf56be12eb721bb8d5aca5ef62a3ddbb52e3d4bc83439c3e8c7d1f9f1dc1",
    ("src/experiments/instruct_config.py", "6a9ba855bc9ba4debb4f1a9fd6ec7964458debb6b396c31b31802a7d59a34f8b"): "f4c79a9890055de3f5fa23eeab71a720081535d75cf2f6d244ac23163637a97a",
    ("src/experiments/mmlu.py", "87963d70654e0e8cafd810eaca8cd1b43004cd431c29a39cce457b2644c5cabe"): "05f8f987fc3231702721963e544987823f36e79be0610401b8c8c9a9a1f8dea8",
    ("src/experiments/mtp.py", "b6d5156c4f2ffd8f17c4afee308894c335aae9dd4b93e35f5c19868a7f22aa74"): "2fbefc08e0ca538a0c9961c6e70c3c9d47ad084cecccc83de75ad60f8ec0ca8c",
    ("src/experiments/mtp_unified.py", "8fb679a90c22b2f9064f44c6e83da8c8a069a2428a158709a5ede1c036d4e6a0"): "89df47685cd3047eba5faedb360ab1288a54c66544500d69d6affca94b4c53b3",
    ("src/experiments/provenance_prepare.py", "7c3ec9e9605cc0d090975eeda784387e783b7f933d350aae5cde2c92a39a8c97"): "d6efec180ace72aa7ccee8c3508ba2e86b3f665cf8400c40e7ee29d47d046522",
    ("src/experiments/qwen35_cap_compare.py", "4042dfd2c411c71f3fabccfaa413e245f4f7cd8a6484fe2eb3eba79cf72b5d09"): "832c6538dcb89541d63d40291f29f56376f213b1c2521089698414abe84afadb",
    ("src/experiments/qwen4b_drafter.py", "0a265933286260e59d58ad53ddbdec4be4cb1d6d7388dda0d834d183eb39f5de"): "fad7fb37c9643936c614aa4a9848d8527c1e78af0db2eafe5d743faae07a896b",
    ("src/experiments/qwen4b_drafter_data.py", "8e9b0a264f39fa28b3715ab73ea7a186a5056b3ae6c5435262c632998413da80"): "f7fc7e60faea68b5be3b26aed58f4595735ef0b3c340c07d884e95866690508e",
    ("src/experiments/qwen4b_drafter_eval.py", "9c8b3855447fb562341ee4d2bd34e717b7f7b65a017e4514c8d606675d2b91e4"): "7e0151445f41078d5fdcc3ba0b0356eb466bd263eb61cc1196224899b4db6256",
    ("src/experiments/qwen4b_drafter_train.py", "d4b6cdb436b9510f15e83d746eb01fab4e8e0816388ef913873d7533ef2c6686"): "ca29df9d93ffe15349c34c09daa2f1972d7f1fa6cd127aa52ce1a1d3ec20d48b",
    ("src/experiments/qwen_adapter_run.py", "e681b012acab108a5b63b1265a62a47e9543ae55d64db94422702c7098ee2430"): "2a0ed2527d528347925372de23cf22829f482c9f3d536c38b685df39b156dbc3",
    ("src/experiments/real_readout_eval.py", "786b773e1805715ea8e81e26d1953dbab20dc8f6a5466672dcc25da95ed7858d"): "a9bccdbee24dde37c802b7ff6f2ff96e79e4930411858869372abc2032296c7d",
    ("src/experiments/real_readout_gsm8k.py", "d90a322a81596705007f8b5ba3f9c65270281e94c548423d57a02630e5fd53d5"): "5ed268ecd59712dd1ebd101a0617a56c25902f278f394950e994d66d45cb1371",
    ("src/experiments/real_readout_humaneval.py", "17ba7e8f977df0e78235c1858acf3d01aaabb9db4fc18e3b6f3356192949bcd0"): "4446dccfef9f617e6ebc0a12d0fb8a0c83381ce4c37f13d289ebfbba875a7f16",
    ("src/experiments/real_readout_mmlu.py", "15b1681f1cda3b2cfa2b5c142ce850a5b822947d746e02b0e2807acbde50a58b"): "3a586dab5dc5b2aebfca7b9d81d14a837349fac8623fd6f5b62cfeffb415ee4f",
    ("src/experiments/real_readout_run.py", "8d49d1d3840ebb5acf13ca164dda10bb8a96ad6399cf701daece714c85c36405"): "6485e7c524274ad426ebd790ecb91308a1a84618095ae702efff4f6708c54ef0",
    ("src/experiments/real_readout_serving.py", "98db972f912e918bd4f38f796021aa2abd787a6860cd701af6904073eed42691"): "7283992359db5682214280a4ae3e3cdd8f00fcc4b06e592f6fd3489e7f254ad7",
    ("src/experiments/sampling_audit.py", "62b93c13908f7968ac489758714d288cf04ae4fbdbe3dca9dbf742793cb01dc6"): "084c001c2c310e6aecbfbca7d579c0ef704e8382b009af79d682d8631c2e5814",
    ("src/experiments/sampling_trace.py", "ff0a95fe2b4fb016e8b1bd6521d381d8758e34f749be53f44b3206c3f4d77542"): "e6fe5ec076a073558e8765654ab185e9e12d9f7bf07b64f934b56d4290e60c2d",
    ("src/experiments/sched_config.py", "d9f964976353e2ea839818a889e3d6563d04d1637b41d42b10a345b6072d4bd3"): "21b73cad2cdf69cefa4aa366d268c4385e49d5e2f5731f156ccf0b11302a1925",
    ("src/experiments/serving_benchmark.py", "52591da083f83e52433010db6d9ae9a1418ff1168186ff743cdeb333aaea57bf"): "cc6c7ba07baf5efb929cd803504bffe11b9060d954ab8ba4eb603b4b914e8227",
    ("src/experiments/serving_graphs.py", "7550a3b4375dfc672c9b3a51495996e8aeb5c84d834b2907820b2cb64f641cd5"): "af6f3c734242de4a3921b3ff761d66736540fc4b6098a1063e8bca2f5cbc336b",
    ("src/experiments/spec_baselines.py", "32a7a28d990b363514a7e1ea5f89eeba881ff1faec4a63b28548ff097fc42581"): "03a1b4377b9cac87f0fe041b63273082e018032ca36aee8c8c003018efac8454",
    ("src/experiments/teacher_cap_probe.py", "d691a7dc5ff07240d2528d8368b52adde803d7f873817c43622e553f86cce4d8"): "44f87b33ba238cb434532eddd2c272ac79361778353e45e1233361bf7803c427",
    ("src/experiments/vllm_sps_config.py", "a281ca9ca5f826ae4e753139310b0951ffea4f7960734ca2645c0c395e38c575"): "34e526a7968fd7cbf9fa8e0f0da3a5f044fc8aafbc6bc2e14abad30a2172da76",
    ("src/workloads/adaptation/ar_sps_common.py", "53215b380ffc954820146b7a991d29018baf12f215295678611cd6c27b8eff1a"): "804b8ab4e63c2a63c697a818ea3905a07c7d9bc295a9f608c6247312fb1a256e",
    ("src/workloads/adaptation/ar_sps_evaluate.py", "0ff9aea92d0856a7b7d2504d8da81a5139ac8031f81998de9d93878ed6c63f30"): "6d661eb532340e141427b8a00f31ce3c721fc816f63223cb4d596eb6341e9457",
    ("src/workloads/adaptation/ar_sps_report.py", "3e33360ff2ff552b57d6420a62152041131c07ddb98b1dd10a0860d70ec4f971"): "508f7bfc0c2d4a8f35b6ba1ee4fd3dfabfaf04daaac41fdba8d584adace510d8",
    ("src/workloads/adaptation/ar_sps_run.py", "f0471b18e792ac28b63bc35c6e617daa5e91eae996f42af32760496d8c850958"): "40cf420507f94a9cb15c5607094ebe72338bf24b96262e3997f0c452168e5b18",
    ("src/workloads/adaptation/ar_sps_train.py", "3857a11c25c6def1a12997132d4afbbb91e0bb0b2e2d0275117a74695eeb7464"): "8ec4bd5d485bb4edfa5a5583eeeb3f162490f1d61c986fe389b4eb3d8456a345",
    ("src/workloads/adaptation/base50_config.py", "9090a64d6d0ce253e48d7685724fff2e227d66064fd2c525a5958eeab89b4b1f"): "ca1dc73f50ea1332bc49c0fb39d2d1d351e1f78511db2c5958f4d0a59be04d5a",
    ("src/workloads/adaptation/base50_run.py", "a49b6a7679c461e8743776307caa6b669fa7cbd097456c8b379296542f2933fd"): "8a926e14d3d584cc2126d0e35fd17ff667435bbddf290856c35633dba5e79250",
    ("src/workloads/adaptation/base50_train.py", "ad2ad7d4472b56dcc144f473618e7455a83d1e2d6b81f38d03a6f666ec3db2d8"): "eb79f69eabfa8f36a7e22536fd3aa86cced8770a7de031edaaac6a7ea723d977",
    ("src/workloads/adaptation/qwen_adapter_run.py", "75103398dcf12777cea54ac60bbf0ae5426b7a01626edea2e594652273cea7f1"): "169ae0d15c74b2324a838d54979cf86d9d6474fdc5bf82f0d67b022cc1a0a8a6",
    ("src/workloads/adaptation/qwen_adapter_worker.py", "9440b013d111dfba045692222d7b586ad40f697e65f3c4a8a62d90f84ce5074c"): "ad9926615ae1e1ae3238ea5960dda2c75d85ace783aea0dd2604cb301389ea3f",
    ("src/workloads/distillation/dflash_kd.py", "aede1e30157eaa8210c1728fcd778fb825c698aa5cfac521d9507d80cbdb92f7"): "87acdd7ba526db16d23163bd71a734f66c47400bb0f81e55f4e95d4187c755fa",
    ("src/workloads/distillation/dflash_kd_train.py", "54240724c98410a32abecaff065d05b90e545c8c5e86d7961e432fdcc372aa75"): "c55a6819a231bb19981687bcb691803df0616456904dc3e662f5b59e0be8e310",
    ("src/workloads/distillation/dflash_mask_ablation.py", "dc8ef31b9d08ed71bcb1c04559a5219ddd85d60bfd585dce828a45f347ca2d12"): "83d9e1f0448858b963db5cc0801f641f4f85d21d09b3987762c74e09e287ec5b",
    ("src/workloads/distillation/dflash_repro.py", "5026f597a507cacb6577afa43b07fb3137ce558da457732efad49a19a2da8cdd"): "c6d0a1e30ac5859f2f0f6a2c572e53acc4512fac0962a1fc0053e04b7cd54db6",
    ("src/workloads/distillation/dflash_repro_eval.py", "38722cbe01f569bf933a52a32b4cbb12d5d2b744d91368983aef2da2b0d0c069"): "17fb81dc97b11a35941aeafb175a00e5e6130931bd71caec9122bb82ed2b83fa",
    ("src/workloads/distillation/dflash_repro_train.py", "ec580296b27ad1fbf5664fed12d1a77799b367f3cdd4d47440112d45642bd236"): "3d838ef07c5e0a345999059bd9181571d58a873c6107b1eb7ecffc5606b244b0",
    ("src/workloads/distillation/qwen4b_drafter.py", "7e4cbbc2d364f1635cfef209cf0da0554ffa98624b69b3371a8cb513b91cf2d1"): "97eee8da4769b6268c5548224400f10f3267cf043a820d07c0e7731e9cf56e4e",
    ("src/workloads/distillation/qwen4b_drafter_data.py", "4f0087afb9569bf7f2fb34523a36a447749acf1c3e666058d258e519a43e4b5d"): "abbed60acc831f5bc11f08bb42b060ea3fbec81950ef55e84864e192c9fe5960",
    ("src/workloads/distillation/qwen4b_drafter_eval.py", "70a072e8e982768ec675e3964e04282df62844d54ad3e81e6a94a6c4c5cb3007"): "1534d6b9aca6251a9415229be467bd3e37fa652ac6dcf516e0e32a44ea36d21b",
    ("src/workloads/distillation/qwen4b_drafter_train.py", "ee7e2dc987c489813ee4934748f7b67b56fc762a23fed37afec8e146b6d6920a"): "d11762dcd5668935bfca494ea348ed73200fc70372894b94e7a1c77d9f788dc2",
    ("src/workloads/evaluation/mmlu_run.py", "ab5e607df644f05f8013fb61f69117800ab8e81d0edcc354d732ebc25aa5396d"): "1475f2666520163a19ba3d940a5d08aee3696c54aeeb58f4c804891217d565ce",
    ("src/workloads/evaluation/mmlu_subset.py", "c771bdceceec8d091c105bc16eb8111c36ec218bdae0c54a8b8ddace197ef13d"): "9e97dd3834313f939c6531b50234da5714bdf66112c1fd3228a846bc9f3f5dbb",
    ("src/workloads/evaluation/qwen35_cap_compare.py", "eb7fc915b679087509ede5ac15a4c4cdddb1e901870c7a59cf4d035dbd46cbd5"): "368b7c39020d6f4dda4a336ec92f8dc221e24ca187bf6539d9aec3dea2380e12",
    ("src/workloads/evaluation/report_mmlu_stem.py", "8dbd07a27a48743c3d755cc64a7c4fd674965d4dc0e52f1dc9b27ebbd234a01b"): "087b63226664a1efef8a233814f89a8a3de1e7f44245da8924263f8adf16a6b6",
    ("src/workloads/evaluation/teacher_cap_probe.py", "71cc421a935287cfbfd18bf9ac19f1a0423508996a5073dfc949446c27582e65"): "5bb50eb67bff7ec297e740ea8bffdce6d220f9bf5051cc47d67a5755634f82c5",
    ("src/workloads/instruction/extend_common.py", "dcd49bd42ce7a1f80a44dbbc85430861104f5b7e832503f485fd5112a4021729"): "fe8708641d50e03ebb986e46b6a34d0a25a3275299ea53b05a1c95778e4132b5",
    ("src/workloads/instruction/ifeval_run.py", "83a1a1fb90cd5d06b4ed45e498ac1c269ff66bd62fcec4add6d91f80ce9fae44"): "ef2d88a52b685bbf15844ee2e6d7c189afff5044d8c6e9bb2526901e9d9904bb",
    ("src/workloads/instruction/instruct_common.py", "30d045ce21d978bc00d763ef51c9f78aaa14d79d613bf9ad274f891787221bde"): "e51cde4cacb8d0d9523731fcc36cbec9ff2acbde7ebb4bf5ad74344ebc6e2ab9",
    ("src/workloads/instruction/instruct_evaluate.py", "941f452117a82b527b91b452c961acee41559786968f8ed735e56e4a38cf3f86"): "be75218314b57bedaeaedd62a1cb87c9c6762f6aa8b7ef2cdaf425111b3b6897",
    ("src/workloads/instruction/instruct_run.py", "008408d1ca942d205c930578fcc480a8fac792edd1c20fe11a2c29497ac4018e"): "9eba287e2ad1fa156244e1c8893e1d1e024822c9f01da7f2f3022f5151db871d",
    ("src/workloads/instruction/instruct_train.py", "1513b78d32bb1eb5bf46f846265969862307b601010c28435e62ce78a16ff3f2"): "9500513fb7b75139ea36cb0afd456289f57290ccf32b3cfd8ae5ee82ef53180e",
    ("src/workloads/instruction/sched_common.py", "f716c4d7d237e53b8f365c290fa5d4f25e16263b9b3786aa1afb28c086f9864d"): "279065fcb9c270eadbe781aba2d0b6dcd7e28aff3d1d7bfced933793640c3af8",
    ("src/workloads/pilot/pilot_run.py", "6e704bbbd29cf8f4a72d00a7d9a9afa274c16da1f8bb55efd2907bc7899e1617"): "592807bca5019985e5723b3c97ebe7bd6c36f1ead76b717e630b903bac192bf8",
    ("src/workloads/provenance/provenance_evaluate.py", "126fa6e7667b13e62a90d895efa2b51f88e3aabd97f2e1cb0a018f4fa64ed682"): "e90d039f6d46e02e87abd91fcaea5cd975534ac3ba4797ef73c297c5345e34a8",
    ("src/workloads/provenance/provenance_prepare.py", "23be760d8b480207c23c097b89578604f5b2c289d7a5dd5d7efef5794c3622e7"): "1ee26d8274b6171cbdfc14a973d33c90d95f7ade888392831badca58127a8a27",
    ("src/workloads/provenance/provenance_train.py", "6d4fe182677a9bebe5d755f2730172313d7b6ee4938644e6a75ef856d8b546e1"): "1f01b9efcf490ee730781edcccdc21f385560b8018c82e1554d14103a51ae938",
    ("src/workloads/readout/real_readout_eval.py", "3fbeb9d201b177e76dc03a151d1909faa08b73ad5c71adeff12508cb834136f2"): "1e90ba6331dbff28c930f3eb32646ed9ea319256b11083fb5b795f467768864b",
    ("src/workloads/readout/real_readout_gsm8k.py", "62d1d4e675c9210e47a6efbd14230e65f76a1a69a4f7343084e173dafe07bbbb"): "2fd3b9337145a69afc29c788c2b1d3e86ac0455a215becc3a34a3196c0db44d3",
    ("src/workloads/readout/real_readout_humaneval.py", "f2b3273e56f600e1fea35b76d0483281f2791b33a5f8755f5c9dfeb50911190d"): "8284d2848419985bdc2830349cef7c5361c772fd5d95b1c1a695e2418643de53",
    ("src/workloads/readout/real_readout_mmlu.py", "a0fd125975e13806da4151c7566dd5677dff192d29c0b47635c69c4d66bcd551"): "e686a7040dfb97e0cb3b74cf68d7d4c3dee653b6051716887550cdaca22b5759",
    ("src/workloads/readout/real_readout_run.py", "e1c6f420912c502e31bd047954e15bdfa0697dcac6f7fb53cb84da54e6dd922e"): "2c25fd6d1dc5bcff5db6eba925dae2ffdf5ce5a350808e7987808cbc9fe9bc35",
    ("src/workloads/readout/real_readout_serving.py", "21fcf67f1e737e31a955e610c36cf885f0a7ed9503312ae8caf0b9f09174ee5b"): "c4d65a4f0c7006b627ee6a94a00a85531fd6654ae172d056158854f194a6e367",
    ("src/workloads/readout/real_readout_train.py", "cbe5a4b3dab4d31447ae81fc7e164679434ab621c0158796c07affb26b7c661b"): "e054d73a24c651fb35c25441de6b08275b41e4aee746be6fc210f9b8a515a3fb",
    ("src/workloads/serving/draft_sweep.py", "0dd657ecaf4b0e89c95400b8e9b01db77399c906c9aec2c63d000b70840b6437"): "cbd47a8f72edc0eef7b7c76d0e3183771bdf59332ca5471c4b5aeab9e01bba6e",
    ("src/workloads/serving/draft_sweep_train.py", "15d747c62b7b6debc837b4714a78e321b984cb7cb74f41beba67fffd842c4321"): "d7058c995dd04949cf77d6a8682b0d64b9fe5ada2435358dcbb7188e1dfae223",
    ("src/workloads/serving/sampling_audit.py", "82fbfd839f496079eb2886344c1e079ba713f350245d048f0276cb74de04b302"): "79fad8be3df0f5234baeba5036d315584c2bcb6c89b98b301b9f1034d57e1b75",
    ("src/workloads/serving/sampling_backbone_precision.py", "9b507b6c6981ed1ed24c65250e85588ceeeca1b05b28753c9a30fd12c816fd7d"): "fc016300bdb5bb69a6ed22999931377e8ccb7ec5d054a4ba4b765bddb9247d17",
    ("src/workloads/serving/sampling_cache_trace.py", "27e41495720db4d9809cec007b42075012a4220d6fbd8427d860f3c165f95fb6"): "f860abeca174b22e33e343ea67246c4790da7f4bfe48af39449cdfd3feab56ef",
    ("src/workloads/serving/sampling_precision.py", "ca7178bbe0bf5a510508e9dd41cc23c5f91837f532c851af43968ee278995e14"): "ade8f569031522bd0e1e9a33644f132ec3ec7adafb8c11b8d0a766dbb3b92681",
    ("src/workloads/serving/sampling_trace.py", "cb5509cbeaa26b8af0da7029a9207bc3786c16828a836cb6398c1eb4e550f043"): "698b8f7ececf4775f67b746c8d4ddb30c01a7be3d3e3984f7ea9788cb53593d6",
    ("src/workloads/serving/serving_benchmark.py", "3673352f187c3a29bb454bb9de26b4be182c6500e3b81ca95ddf4e1f61ca4f3f"): "29217e8596d658ec23dc0c9b884568b0e3fa863e71aff3ba17bd71ba7b5620b0",
    ("src/workloads/serving/serving_graphs.py", "931784944ca2f398f1ddb0eb6a9cce2ae5b80439097e73eb0c52cfe2885b9c23"): "f25cf85077c5d38a596f884d5d1b89939140c95e2a57af6a2f4adf5f8f254c5b",
    ("src/workloads/serving/vllm_sps_evaluate.py", "20755589df7b8f75f8f8f3284e6b73c4494a3672e6d1d032024cf8edb9ec328a"): "7f30a8e8f2e3efcf8f3c951d6f19c8e0dceccccf60a47eb1a7500d4e95fde96c",
    ("src/workloads/serving/vllm_sps_train.py", "a024cc92b8bb98c8818d00a1667f491dab8e03bdc412afd114872fad82ec36ee"): "8eccf959f51c19380bdc3cbe568c9847bfd9b40e5ca7fc6968a10ea936438c38",
    ("src/workloads/speculative/mtp_evaluate.py", "b889562ce22c16dddbbb4abb259955a6c91693109c271dff6fbb4892efd03092"): "6b50d6eb997601e28861a1fa191d4008fbd65e2b0f7244625670e1ac1ffc80af",
    ("src/workloads/speculative/mtp_parallel_evaluate.py", "6a7db3d763ca2783214883ae18851baf1057aac31739534bd4029cc00285a80a"): "0d94c4e3b91f476373ea16d68b536c46b156ae94e3b627b4a49994082d087fd7",
    ("src/workloads/speculative/mtp_run.py", "02228b6891543d44f215ecde198636e50154a476c818cb9784f2f92e2b5d1e27"): "57f2713d715400a13da37185dd8d292d458f8e8dc2319cd06516306bac767aae",
    ("src/workloads/speculative/mtp_train.py", "f323e39ed291c135797d20caa6bb7420b5c9a87433ec4b3584f9892e2f302201"): "a7c191b7ce426b214d4a0c79877edb9bb2b32c6fc375f4a11d7dfaec64a5e65f",
    ("src/workloads/speculative/mtp_unified_evaluate.py", "bfcae11c20dff9855e05bdf0edba9ecf709ce82662e9e526fc71fec86177774c"): "0dbe04e136cecf1273c2e091d66e4d370016514611b24f9aa8f4fac51f7c5ce8",
    ("src/workloads/speculative/spec_baseline_evaluate.py", "5f002cc8044cd6aa903fc2a6e1d20e9e877da6bdc1d62179d852a441210c4967"): "f13887531bfa4b13908a137b66a3cb811e9ee47b1d57c112208bebdba4de6a0e",
    ("src/workloads/speculative/spec_baseline_run.py", "eeb5fc8d6ff6d0e55c93132e53b4ac5e7fab0b109502626cc103539348ff7ff8"): "4147697c16d487f988d7d2f639ae6f8d159576c9a338b3cdcf4c323adde882a7",
    ("src/workloads/speculative/spec_baseline_train.py", "f60003cd26c59cb7c927786fe52c47173c7719b408000357eab75b512522f75f"): "d14197f9fe7bbde8b49442f299a78ae40d20b80dd49d0522018bb223f704312b",
    # Previous migrations.
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
