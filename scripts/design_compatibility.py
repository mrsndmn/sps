"""Preserve completed runs across the suite routing refactor.

Each key is the full design hash after extracting the worker registry. Its
value is the hash of the same design at 98fb926. Changes to a hashed input
(including the selected worker) produce a different key and pass through.
"""

_ROUTING_REFACTOR = {
    # 10b
    "bf49201709cd9b2ccbe2184f33a2e3063be91d34f7c22d070e5d8a01cf67115c":
        "6b63ec67fa74e417455a4e44955105dce9247d8dc4649d402f5a815f0d5447b8",
    # 1b
    "884a883c5490fb1042ff8681470ca6bf059a1cb279a877d4911b72900812c37d":
        "344d91c4e7664df9c58cdc44d04fa191886c20f95f3bd271877b1b1e6d899c1f",
    # ar-sps
    "48254eb9915639429409fcbdef85283f8182dca64f6a4e3cc06c967d3debc1d7":
        "58079b749a5ebe8c5d59e041438ef590919a73ecbf5a0a1ba3fa9ea510d51df7",
    # ar-sps-1.7b-ce
    "04371ca18f2d6e0c8d4be98160176abae9a80f34e557e5fce8b01a2f87e46d7e":
        "59b62831e85f49d241e53f541c4d3166b2e5c789d0d9a3de8a0613abb8399fa8",
    # ar-sps-1.7b-kd
    "714353b439af1d0ff9577b23a12a4ec4c019c38e9a853de2640be4619c40f11f":
        "93fb2508e012dc9cdcb2f2107802dc5fea2eade1d136f3c6d3a0f04087dd9001",
    # ar-sps-360m-ce
    "699cc6a63a6c26e20358abd8e3741dbdc0ceb18c2836706e70c727b6260ebd5b":
        "3e4604921c3e8c3b2dadc3def1bae1d772d1d4420988d8dbf9a6ec603ae90b99",
    # ar-sps-360m-kd
    "ed599e77f475f5996ee86f91203ac00160f408293ec3d3779c86e445cf7ddd4a":
        "5f703ff776f4354784bf81546e0be0d0955df7f449d8b2183660ac523dfd01f5",
    # ar-sps-kd
    "82e70173d1655a35cbe9c2d2086cf503e2ae6cad2d67b3cf84762f221cde82c2":
        "0c7f96387ea7bc814c1fb13aeca79ecd6f99d76fbaadf373f658b1678f88f149",
    # ar-sps-qwen3-0.6b
    "026ba11018fcf29da4b19eb49a62c82f62083391430ad523dff37ab641c9b442":
        "065388611d07ccba134b38ae99df13ceee1ad3d46a25159289364e1878f6cc67",
    # ar-sps-qwen3-1.7b
    "56f1768e4416699320c0a8719fff873499c21fb1a44d646df709e4c1e47212bf":
        "b31e4559f6fc2603287161ea696b299304643e662f8f66c4db0150d451476b41",
    # ar-sps-qwen3-1.7b-instruct
    "155d4d4703051c1acf356e6baacf53ec66c4b13eb5088767fd09b5f3d95f7bc5":
        "8386a4caf4ed6ac6c20a8eff5c71e01b5f72b5abeb5c5ac6d67158125e3a30b1",
    # mmlu-full-resume
    "d9360d9b5e1bb2eaec1c5aa9b520d994866aa9aa08a3a730eb7d27d22ca8c1aa":
        "60d5cf5718586eab81588845111dc96e6ccefbf36c33fc847f542c4ac1af8e9e",
    # mmlu-generation
    "44f3c5fa6ae3444c1d9fb14096011c40c011b65b20469e463fb4d711d8dd8f8a":
        "f37b2a2a4722048d235d5136b77361e236ec635b6d91451abd29896a0eb06776",
    # mmlu-stem-generation
    "0817d4a68ce7f40c8d7a160ade1818526026103e804c785d53983a8ed4b64cd4":
        "5589783df10841467856827e573ff377de20397a16ca6560c7661bd37d7c312e",
    # mtp
    "151c1dfac51ac077b96cb252f1f7faa50a49892255285f258520150956a527ff":
        "dc157d2767c66adeed12e31d5b2c25d8114e75d042ee73a21548b2cada5fbc25",
    # mtp-parallel
    "7d94e05ced404c4b558b9364665d96d0cf857dfb189ddce80b9c49f5d216802f":
        "f0bd5cede0e81ca693ed39618879bde412bf0670c1bd59b249dc5cd8c82f337a",
    # mtp-unfreeze
    "3da15d1df868582b06ed99bd502408b19037e6652e7d84751b2b158fd8bff839":
        "f0bd5cede0e81ca693ed39618879bde412bf0670c1bd59b249dc5cd8c82f337a",
    # mtp-unified
    "e2eb16143da1a839d1cecf90650e3e819639e832465dd09347c59e84d96c4813":
        "a0c24f175df79cd6ab5ecf3a08159018ca7171c41ff9154cee2651ddebe93087",
    # spec-baselines
    "7b27e2d0e0457f57e37e98cba3d2d4448a67e937009226fb0c22252b0db2c370":
        "c4e978edfe22a6ba2057eac7aea15eb1a92ba5ed2ab6662f90cb376dcc234823",
}


def canonical_signature(digest):
    return _ROUTING_REFACTOR.get(digest, digest)
