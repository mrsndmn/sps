"""Preserve completed runs across the suite routing refactor.

Each key is the full design hash after extracting the worker registry. Its
value is the hash of the same design at 98fb926. Changes to a hashed input
(including the suite and selected worker) produce a different key and pass through.
"""

_ROUTING_REFACTOR = {
    # 10b
    "708fb78879452124ece962a02f83000459a64746b08735472433c1ce23d5c237":
        "6b63ec67fa74e417455a4e44955105dce9247d8dc4649d402f5a815f0d5447b8",
    # 1b
    "4780e1816b74ecdf53e3e98bc439c1e5f775274129ebcc5df958a0f56f49169a":
        "344d91c4e7664df9c58cdc44d04fa191886c20f95f3bd271877b1b1e6d899c1f",
    # ar-sps
    "8e98fe3826887078a65c0481b319c47272fc93f60af0aaca38f483ad2685f4be":
        "58079b749a5ebe8c5d59e041438ef590919a73ecbf5a0a1ba3fa9ea510d51df7",
    # ar-sps-1.7b-ce
    "fb264e3d937a2cb6ab33573a307b921703b3e92479ea36f72ad46fffab19b8c4":
        "59b62831e85f49d241e53f541c4d3166b2e5c789d0d9a3de8a0613abb8399fa8",
    # ar-sps-1.7b-kd
    "3333a119a3cb61a16dab9df76f80fff670b4e2ddcc3300f97150fb6dc316c1a0":
        "93fb2508e012dc9cdcb2f2107802dc5fea2eade1d136f3c6d3a0f04087dd9001",
    # ar-sps-360m-ce
    "eb9b78b0402a2731ab3e836275208a2e169680942c74b3db14ddf9d1024059c1":
        "3e4604921c3e8c3b2dadc3def1bae1d772d1d4420988d8dbf9a6ec603ae90b99",
    # ar-sps-360m-kd
    "52df18a1027a64ec69b3f4fe3288bbbf5503adc16f0d2615b70ef1ea40836140":
        "5f703ff776f4354784bf81546e0be0d0955df7f449d8b2183660ac523dfd01f5",
    # ar-sps-kd
    "c32e908459bbdad544ee727fcef8b8f2c61248850eead96308f1ae96892457b8":
        "0c7f96387ea7bc814c1fb13aeca79ecd6f99d76fbaadf373f658b1678f88f149",
    # ar-sps-qwen3-0.6b
    "b3321fbf1e37f1338109617d8603e2c9f817b0c0180ad779e104e6770e28793e":
        "065388611d07ccba134b38ae99df13ceee1ad3d46a25159289364e1878f6cc67",
    # ar-sps-qwen3-1.7b
    "22aaf5d40e40e1d2fcb326b551e378578fe24ce1a691533c3207c3de137d7a37":
        "b31e4559f6fc2603287161ea696b299304643e662f8f66c4db0150d451476b41",
    # ar-sps-qwen3-1.7b-instruct
    "b4c57ba537b3773faf330616e29f87a53c6cd761efc2d542d42172a6c8001818":
        "8386a4caf4ed6ac6c20a8eff5c71e01b5f72b5abeb5c5ac6d67158125e3a30b1",
    # mmlu-full-resume
    "551b4f401d2e2792801191a0cc123364ab33b232d3d568d3a1151fd4667b2171":
        "60d5cf5718586eab81588845111dc96e6ccefbf36c33fc847f542c4ac1af8e9e",
    # mmlu-generation
    "f75abddb2d550845b5af96967dd178411aad8642f54aed3c6ab5479312a11197":
        "f37b2a2a4722048d235d5136b77361e236ec635b6d91451abd29896a0eb06776",
    # mmlu-stem-generation
    "35af76dc4ce9530dd45845f6ee86764cc490b1bfc5382f98e84f164bdbde3742":
        "5589783df10841467856827e573ff377de20397a16ca6560c7661bd37d7c312e",
    # mtp
    "d598315c6425eaaf8c530fbe4bd3cbc5bd9a348d20c3c49e90796d0dac4c85fb":
        "dc157d2767c66adeed12e31d5b2c25d8114e75d042ee73a21548b2cada5fbc25",
    # mtp-parallel
    "3baa38ba32a69e63d7874a85d3afbd24bf5fba4e7c3151c583a6c9ea03c10c58":
        "f0bd5cede0e81ca693ed39618879bde412bf0670c1bd59b249dc5cd8c82f337a",
    # mtp-unfreeze
    "dfb0ac0226b87cddb13670b75deb6a79958278ac7911295d1c1d823cca9ff2d2":
        "f0bd5cede0e81ca693ed39618879bde412bf0670c1bd59b249dc5cd8c82f337a",
    # mtp-unified
    "7cae2078af58e3a9ef84c82cbf3b375339786eefbdab63329f9565682929ade9":
        "a0c24f175df79cd6ab5ecf3a08159018ca7171c41ff9154cee2651ddebe93087",
    # spec-baselines
    "62da28b20c5ec0a0605ada1df9b4230935c064e3f0dc1ec873436ad3fc6b7516":
        "c4e978edfe22a6ba2057eac7aea15eb1a92ba5ed2ab6662f90cb376dcc234823",
}


def canonical_signature(digest):
    return _ROUTING_REFACTOR.get(digest, digest)
