"""Preserve completed runs across the suite routing refactor.

Each key is the full design hash after grouping the worker sources. Its
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
    "7b4d2e242eb13635ee018252375dcfbd578e6a1791c7b98c268b19714d670bf2":
        "58079b749a5ebe8c5d59e041438ef590919a73ecbf5a0a1ba3fa9ea510d51df7",
    # ar-sps-1.7b-ce
    "a2c9990a4f0edf586eba489f4a4c3a16ac19e5284e5b019157d499e0c851aa6d":
        "59b62831e85f49d241e53f541c4d3166b2e5c789d0d9a3de8a0613abb8399fa8",
    # ar-sps-1.7b-kd
    "1fec6aa01b4976a3c58f9251cbb9d3d42d3908447df1f4c6db1650c462d69c1c":
        "93fb2508e012dc9cdcb2f2107802dc5fea2eade1d136f3c6d3a0f04087dd9001",
    # ar-sps-360m-ce
    "3cb5da73e69b13dc87a558265e3fc93ca3d0fd52f355f7af56101de27b297837":
        "3e4604921c3e8c3b2dadc3def1bae1d772d1d4420988d8dbf9a6ec603ae90b99",
    # ar-sps-360m-kd
    "7b28210c915f2893b8c5ead246322c649d027850e7e46df867583968c9b83b04":
        "5f703ff776f4354784bf81546e0be0d0955df7f449d8b2183660ac523dfd01f5",
    # ar-sps-kd
    "c2fa9a68ac33dc67ffd89a32544c247df2e68d6fe6822ddc016b06977426ef5b":
        "0c7f96387ea7bc814c1fb13aeca79ecd6f99d76fbaadf373f658b1678f88f149",
    # ar-sps-qwen3-0.6b
    "85abb46aa8710ecc7d34d06c37eb83fa2dd682e73c5a2a56a701330c6fd758d6":
        "065388611d07ccba134b38ae99df13ceee1ad3d46a25159289364e1878f6cc67",
    # ar-sps-qwen3-1.7b
    "0f14ec0a5a0c5199bdb3b13ac1cd8846e308acec32638386703d1156d176bd44":
        "b31e4559f6fc2603287161ea696b299304643e662f8f66c4db0150d451476b41",
    # ar-sps-qwen3-1.7b-instruct
    "92925c1d73573b689949a4b1e77d1fe806825840d328f764b4b3d72f483b761e":
        "8386a4caf4ed6ac6c20a8eff5c71e01b5f72b5abeb5c5ac6d67158125e3a30b1",
    # mmlu-full-resume
    "e497481b49b148f03412772810891cbb1665e6d75754b94655470ca80e6d7f11":
        "60d5cf5718586eab81588845111dc96e6ccefbf36c33fc847f542c4ac1af8e9e",
    # mmlu-generation
    "e6a05cc95df92a6839213889cf284ab522abee2f21a9b137f9f6d6d1dd85d5da":
        "f37b2a2a4722048d235d5136b77361e236ec635b6d91451abd29896a0eb06776",
    # mmlu-stem-generation
    "b0d0412d51b1e5f1a3467d7da42ea0d50823bbd227883e13e3f4773bb881c167":
        "5589783df10841467856827e573ff377de20397a16ca6560c7661bd37d7c312e",
    # mtp
    "281999e81c746937c4c29f491cffa0bfca92c899ecff54a6d6dac42302aa438e":
        "dc157d2767c66adeed12e31d5b2c25d8114e75d042ee73a21548b2cada5fbc25",
    # mtp-parallel
    "8a9e31eecc7e37fe8c0ddf59cc4d59f8aa9090c83d413ef9391ae96a1132cc38":
        "f0bd5cede0e81ca693ed39618879bde412bf0670c1bd59b249dc5cd8c82f337a",
    # mtp-unfreeze
    "41138ffa1e72fac779db064f9fadadc01ebb011b3e7a485796f87f6a3770e382":
        "f0bd5cede0e81ca693ed39618879bde412bf0670c1bd59b249dc5cd8c82f337a",
    # mtp-unified
    "5fb04d5e9e28708a650318e63949bf2a9c5473a394d8e0f166aec1f9432e247f":
        "a0c24f175df79cd6ab5ecf3a08159018ca7171c41ff9154cee2651ddebe93087",
    # spec-baselines
    "22ef2d1587b97f4f65a79747e5541b1ce805381d31c65a5ee6747e785a480b4a":
        "c4e978edfe22a6ba2057eac7aea15eb1a92ba5ed2ab6662f90cb376dcc234823",
}


def canonical_signature(digest):
    return _ROUTING_REFACTOR.get(digest, digest)
