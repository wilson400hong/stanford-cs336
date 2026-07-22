# BPE: Training & Tokenizer — Notes

Concise summary of the optimizations and correctness lessons from building the BPE trainer and tokenizer.

## Training (`build_vocab_merges`)

Goal: OWT, vocab 32000. Went from **>1 day → ~10 min**.

**Core data structures (incremental, not recomputed):**
- `bp_to_count: dict[pair,int]` — corpus-wide pair frequencies.
- `bp_to_pretokens: dict[pair,set[int]]` — which (unique, int-indexed) pretokens contain each pair. Lets each merge touch only affected pretokens.
- Pretokens stored once, weighted by frequency (never expand duplicates).

**Optimizations, in order of impact:**
1. **Incremental updates** — after a merge, only adjust pairs that actually change; never rebuild counts from scratch.
2. **Lazy-deletion max-heap** for `get_max_bp` — replaced O(#pairs) scan each round. Push a fresh node per changed count; on pop, skip nodes whose count ≠ current. Tie-break = lexicographically greatest **bytes** pair.
3. **Level 2 single-pass emit** (`merge_bp`) — one walk builds `new_bt` AND emits only the pairs that change at each merge site `(…L,x,y,R…)→(…L,xy,R…)`:
   - die `(L,x),(x,y),(y,R)`; born `(L,xy),(xy,R)`.
   - Accumulate into a per-pretoken `delta` dict so **transient pairs cancel** (e.g. `aaaa`).
   - Replaces the old `Counter×2 + set×2 + bt_to_bps×2` machinery.
4. **Count vs membership are different questions:**
   - count = deltas compose → `bp_to_count[bp] += diff * cnt` (weight by pretoken freq; skip diff==0).
   - membership = presence, NOT sign → remove only if pair absent from `new_present`; add on `diff>0` (idempotent). A pair can lose occurrences yet still be present.
5. `.pop()` not `.copy()` on the affected set; parallel `str_to_bt`; delete dead pairs.

**Deferred:** int vocab-ids instead of `bytes` (pairs → `(int,int)`, cheaper hashing) — `merge_bp` is now ~50% and dominated by `(bytes,bytes)` hashing. Gotcha: tie-break must still compare bytes, not ints.

**Bugs caught:** multiplicity dropped in count update (`-= cnt` vs `-= diff*cnt`); membership removal keyed on delta sign instead of presence; missing `new_bt` write-back; `use_cache` default leaking into grader.

## Tokenizer (`BPETokenizer`)

**encode:**
- Split on special tokens with a **single-group** regex `(A|B)`, tokens sorted **longest-first** (overlap correctness). Whole special-token parts → their reserved id directly (never pretokenized).
- `encode_pretoken`: **rank-based iterative merge**. `merge_rank = {pair: index}`; repeatedly merge the lowest-rank adjacent pair, **re-scanning each round** (so composite+composite merges fire). Stop when no pair is in `merge_rank`.

**decode:** look up each id's bytes, **concatenate ALL bytes, then decode once** with `errors="replace"`. Never decode per-token (a multibyte char can span tokens → U+FFFD corruption).

**encode_iterable:** generator — `for text in iterable: yield from self.encode(text)`. Bounded memory (relies on line-based chunks; pretokens don't span newlines).

**Bugs caught:** greedy left-to-right merge ignored learned order AND couldn't merge two multi-byte tokens (must be rank-based + iterative); `b[i]` is `int` not `bytes` (use `b[i:i+1]`); special-token check applied post-pretokenization (fragments token — must check at part level); `None` special_tokens (normalize to `[]`); `re.split` with per-token groups yields `None` parts.

## Verification oracles
- Train A: `{"low":5,"lower":2,"newest":6,"widest":3}`, vocab 261 → `(s,t),(e,st),(o,w),(l,ow),(w,est)` [tie-breaks]
- Train B: `{"aaaa":3,"xx":5}`, vocab 258 → `(a,a),(x,x)` [multiplicity]
- Train C: `{"ababab":4}`, vocab 259 → `(a,b),(ab,ab),(abab,ab)` [telescoping + multi-round]
- Encode: `[(b,c),(a,b)]`/"abc"→`[a,bc]` (order); `[(a,b),(c,d),(ab,cd)]`/"abcd"→`[abcd]` (composite)
- Round-trip `decode(encode(s))==s` on multibyte + special tokens.
- Full `tests/test_tokenizer.py` passes (tiktoken tests need internet / proxy).
