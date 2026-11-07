# CP2 input revision 1

Approved scope: eight exact Spot native close timestamps and one CM funding settlement preceding the native Context. Original raw archives, funding pages, canonical funding registration, CP1 files, historical sources, strategy formulas, parameters and random seeds remain unchanged.

Original implementation 4870310dfa9dd4a51b64aaa149caffaf48de79f9 and original unlock 51a2bc3e15c4a12a87424dc7217278b965c52d81 remain in history. The original run stopped during input loading before any feature or account computation; its output and external diagnostic evidence are preserved byte-for-byte in validation_history/attempt_1.

The revision implementation descends from the original unlock and restores BTC_IS_UNLOCKED=False. Only data.py, the source-registration controller and new synthetic tests are changed/added for this authorization. A new manifest binds before/after hashes of the two CP2 files; the original manifest remains unchanged. A following independent commit changes only BTC_IS_UNLOCKED back to True. POST_IS_AND_ETH_UNLOCKED remains False throughout.

The formal entry must verify the preserved original chain, exact revised-file hashes, all other original CP2 hashes, 32 CP1 files (lock normalized), 142 historical files, a clean worktree and absent output directory. Fixed 42-account matrix, R5 exact comparisons, statistical gates and figures are unchanged. Any new exception stops execution without automatic repair or resume.
