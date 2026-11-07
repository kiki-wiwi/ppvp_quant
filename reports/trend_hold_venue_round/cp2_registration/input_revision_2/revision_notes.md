# CP2 input revision 2: four official CM daily supplements

The user authorized checking the official BTCUSD_PERP daily ZIP/CHECKSUM archives for 2023-08-28 through 2023-08-31 and filling the verified monthly-package gap. Each daily archive contains exactly 1440 native UTC minutes; total 5760. Original monthly files, cached UM inputs and all historical manifests remain unchanged.

The prior locked preflight is retained byte-for-byte under validation_history/preunlock_preflight_2. This revision retains both prior implementation/unlock pairs in Git history, restores BTC_IS_UNLOCKED=False for input and synthetic validation, then requires another independent single-line BTC unlock. POST_IS_AND_ETH_UNLOCKED remains False throughout.

Only data.py and cp2_control.py are modified from the prior implementation. A new input-only binding helper authenticates four exact daily source identities, rejects missing/duplicate/out-of-range/overlapping minutes, and binds UM to its current Parquet byte hash plus the unchanged frozen content hash and historical manifest hash. The frozen R5 content validation remains in place. SPOT, combined CM and Context-filtered CM funding must match the locked preflight content hashes before Context creation or cache writes.

No feature, formula, parameter, random seed, account matrix, cost, gate or statistical method is changed. All 42 fixed accounts and the six archived R5 exact daily/terminal comparisons remain required. No new strategy variant is introduced.

Raw input registration, locked input preflight, synthetic verification and before/after source hashes are bound by the new revision manifest. Prior manifests are not rewritten. The formal run requires the full preserved commit chain, a clean worktree, both required lock states and an absent output directory. Any new error stops execution without automatic repair or retry.
