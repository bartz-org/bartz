---
orphan: true
---

<!--
bartz/docs/notes/cpu-tree-blocks.md

Copyright (c) 2026, The Bartz Contributors

This file is part of bartz.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Blocks of trees on the CPU backend

With a variable number of trees (`Forest.used_trees`), an MCMC step should cost in proportion to the trees in use. The part of the step that is vectorized over trees, the update of the leaf indices, gets there by going through the trees in blocks and stopping after the ones in use. On GPU it does that. On CPU it processes all the trees at once, because three behaviours of XLA's CPU backend make blocks slower than that unless few trees are in use.

This note records the three behaviours, the workarounds we wrote and then removed, the measurements, and what is left to try. The third behaviour also slows down a fixed forest with fewer trees than CPU threads, which is why `CheckNumTreesCallback` warns about it.

Everything was measured with jax and jaxlib 0.12.0. The statements about XLA internals come from reading its source and its compiled output at that version, and may not hold in a later one.

## Summary

| XLA:CPU behaviour | Effect on blocks | Workaround | What remains |
|---|---|---|---|
| A loop with a light body is compiled to one single-threaded kernel | The whole loop over blocks runs on one thread | A two-element sort in the loop body | Nothing |
| A write into a slice of an array is never multithreaded | The update of a block, fused with its write, runs on one thread | Compute the block into a temporary, then copy it | The copy is serial: with all the trees in use the blocks take 1.6 to 2.5 times as long as all the trees at once |
| A kernel over more than one and fewer rows than threads is not vectorized, unless the number of columns is lucky | A block of more than one and fewer trees than threads is several to tens of times slower per element | Blocks of one tree, or of at least as many trees as threads | Also affects a fixed forest smaller than the thread count |

With all the workarounds and good block sizes, blocks made the step up to 11% slower with all the trees in use and 10% to 45% faster with a tenth in use. Processing all the trees at once when more than half are in use removed the slowdown, but the code had become three workarounds deep for a gain confined to forests mostly not in use, so for now the CPU does not use blocks.

## The update and its blocks

At the start of each step, `update_leaf_indices_and_related_caches` in `mcmcstep/_step.py`:

1. Applies to `Forest.leaf_indices`, an array of shape `(num_trees, n)`, the prune left pending by the previous step and the grow move proposed in this one. These are two elementwise selections per (tree, datapoint) pair, written over the old array.
2. Recomputes the cached counts and precisions per leaf at the nodes involved in the moves, which are reductions over the datapoints of each tree.

All the trees at once, this is a handful of XLA kernels over the whole array. In blocks, `variable_length_update` in `_jaxext` wraps the same functions in a `while_loop`. Each iteration slices a block of trees out of the arrays, updates it, writes it back with a dynamic slice update, and reads it back to compute the caches. The loop stops after the block with the last tree in use.

The sequential accept/reject loop over trees is a separate matter: it visits one tree at a time, and stops after the trees in use on every backend.

## 1. A loop with a light body runs on one thread

XLA:CPU has a pass (`SmallWhileLoopHoistingPass`) that compiles a whole `while` loop into one kernel if an iteration goes through less than about 1 GB of buffers. Such a kernel runs on a single thread, whatever its body would do outside the loop. The compiled module shows it as a call to `xla_cpu_small_call`.

The pass leaves the loop alone if its body contains a sort. So the workaround was to pass the loop counter through a sort of two elements:

```python
def cpu_multithread_loop(x):
    """Return `x`, in a way that lets xla:cpu use multiple threads in the enclosing loop."""
    return lax.platform_dependent(
        x, cpu=lambda x: lax.sort(jnp.stack([x, x]))[0], default=lambda x: x
    )


# in the body of the loop over blocks
return BlocksLoopState(cpu_multithread_loop(loop.i_block) + 1, state, ys)
```

This test pins the behaviour; if its first case fails, the pass has changed:

```python
@pytest.mark.parametrize('multithread', [False, True])
def test_cpu_multithread_loop(multithread):
    def loop(x):
        def body(carry):
            i, x = carry
            if multithread:
                i = cpu_multithread_loop(i)
            return i + 1, x + 1

        return lax.while_loop(lambda carry: carry[0] < 3, body, (jnp.int32(0), x))[1]

    x = jnp.zeros(1000)
    text = jit(loop).lower(x).compile().as_text()
    assert ('xla_cpu_small_call' in text) == (not multithread)
```

## 2. A write into a slice runs on one thread

XLA:CPU marks the kernels to run on multiple threads at compile time (they carry an `outer_dimension_partitions` annotation in the compiled module). It never marks an instruction that may be implemented as an in-place dynamic slice update. This includes the fused kernel that computes the new block and writes it over the old one, which is what the plain loop over blocks compiles to. So the update of a block runs on one thread even with the sort of section 1.

The workaround was to put the new block behind an `optimization_barrier`, so that XLA computes it into a temporary with a multithreaded kernel and then copies it into the big array:

```python
state_block = update(tree.map(take, state), xs_block)
if materialize:  # on cpu
    state_block = lax.optimization_barrier(state_block)
state = tree.map(put, state, state_block)
if not materialize:  # on gpu
    # read the new block out of `state`, so that the write is in place
    state_block = tree.map(take, state)
ys = tree.map(put, ys, post(state_block, xs_block))
```

The copy is still a slice update, so it is serial, and we found no way to write into a slice of an array that is not. Its cost depends on the single-thread memory speed of the machine.

A toy elementwise update of 200 trees by 10⁶ datapoints in blocks of 25, on a Xeon E5-2680 v4 pinned to 12 threads, shows both behaviours:

| Variant | Time (ms) | CPU time / wall time |
|---|---|---|
| All the trees at once | 15.4 | 11.2 |
| Loop, written in place | 53.0 | 1.0 |
| Loop, written in place, with the sort | 62.6 | 1.0 |
| Loop, with the barrier | 92.0 | 1.0 |
| Loop, with the barrier and the sort | 78.5 | 3.2 |

On the same machine, with the sort and the barrier, the part of a real step that is vectorized over trees (200 trees, 10⁶ datapoints, all in use) took 110 ms in blocks of 25 against 53 ms all at once.

## 3. Fewer rows than threads are not vectorized

This one is not specific to blocks: it applies to any elementwise kernel over an array `(T, n)` with a per-row operand, where `T` is the number of trees in the call (all of them, or a block).

### What happens

To run a kernel as `P` tasks, XLA (`ShapePartitionAssigner`) splits the leading dimensions of its output into partitions. `P` grows with the size of the array, about one task per 4000 elements for the prune kernel, up to the number of threads. That is the number of CPUs the process is allowed to run on, unless the environment variable `PJRT_NPROC` or `NPROC` is set to an integer, which replaces it; and it is at least the number of CPU devices jax is configured with. XLA does not expose the count, `get_cpu_thread_count` in `_jaxext` replicates the rule.

- If `T >= P`, it splits the trees into `P` groups. Each task loops over whole trees, with an inner loop over the datapoints that the compiler vectorizes. This is always fast. It is the case of any forest with fewer than about 4000 datapoints, because then the tasks are fewer than the trees.
- If `T < P`, it also splits the datapoints, into `a` by `b` partitions with `a * b <= P`. The compiled kernel then goes through a flat range of elements cut into `a * b` equal chunks, and finds the tree of each element by dividing its flat index by `n`. With the prune kernel the result is a scalar loop with a division per element (seen in the disassembly).
- One exception is when every chunk is an equal part of one tree, that is `a * b` is a multiple of `T` and `n` is a multiple of `a * b / T`. Then the compiler takes the division out of the loop and vectorizes it. A single tree is always fine.
- The other exception is when the trees are at least as many as the chunks, which happens if `a * b < P`. For example 32 threads make 6 by 5 chunks, and 30 or 31 trees are fast. We observed this without looking into the compiled code.

The partition counts are, with `t = floor(sqrt(P))`:

```python
a = min(T, t)
b = t
if a * b < P:
    a = min(T, max(a, P // b))
    b = max(b, P // a)
```

For example, 8 threads give 4 by 2 partitions for 4 to 7 trees. With 4 trees a chunk is half a tree, which is fast if `n` is even. With 5 trees a chunk is 5/8 of a tree, which is slow.

In practice `n` is arbitrary, so the rule is: **a call over more than one and fewer trees than threads is slow, if there are more than a few thousand datapoints.** (Precisely, fewer trees than chunks, which are as many as the threads or slightly fewer.) With multiple chains the trees count once per chain, because the chains are another leading dimension (2 chains of 4 trees are fine on 8 threads, 2 chains of 3 are not).

### Verification

On a Ryzen 7 3800X (16 threads), pinning the process to 4, 6, 8, 12 or 16 CPUs, and on a Ryzen 9 7945HX (32 threads), pinning it to 20, 24 or 32, we timed the prune and grow kernels on 1 to 48 trees at once and compared with the rule above. The partitions XLA picked matched the formula in all 267 cases. Which cases were slow took two corrections to predict, each found by a prediction that failed:

- The first version of the rule ignored the divisibility of `n`, and got 3 of the first 100 cases wrong (n = 2 000 000 on the first machine).
- The second version ignored that trees at least as many as the chunks are fast, and got 31 trees on 32 threads wrong, for both `n`. It was right in the other 165 cases it was applied to, including 30 trees on 32 threads.

The numbers of trees tried were not all of them on the larger thread counts: 1 to 6, 8, 10, 12, 15, 16, 20, 23, 24, 25, 30 to 33, 40 and 48, or a subset.

| Threads | Slow numbers of trees, n = 2 000 000 | Slow numbers of trees, n = 2 000 003 |
|---|---|---|
| 4 | 3 | 2, 3 |
| 6 | 2, 4, 5 | 2 to 5 |
| 8 | 5, 6, 7 | 2 to 7 |
| 12 | 2, 4, 5, 7 to 11 | 2 to 11 |
| 16 | 5, 6, 7, 9 to 15 | 2 to 15 |
| 20 | | all tried from 2 to 16 |
| 24 | 2, 4, 8, 10, 15, 16, 20 | all tried from 2 to 23 |
| 32 | 5, 8, 10, 12, 16, 20, 24 | all tried from 2 to 25 |

Time per tree, in microseconds:

| Machine | Kernel | Fast cases | Slow cases |
|---|---|---|---|
| Ryzen 7 3800X | Prune | 1 to 70 (median 14 to 24) | 118 to 840 (median 150 to 540) |
| | Grow | median 175 | median 750 to 950 |
| Ryzen 9 7945HX | Prune | 0.4 to 16 (median 3) | 110 to 195 (median 130 to 150) |
| | Grow | median 55 to 60 | median 230 to 270 |

"Slow" is defined by the prune kernel; XLA partitions the grow kernels differently in a few cases with 2 trees, where they are not slow.

The number of tasks against the size, for the prune kernel on 24 trees with 32 threads:

| n | 503 | 1009 | 1361 | 2003 | 3001 | 5003 and above |
|---|---|---|---|---|---|---|
| Partitions | 3 | 6 | 8 | 12 | 18 | 6 by 5 |

Up to n = 3001 the trees are split in whole groups and the kernel takes about as long per tree as with 32 trees; from n = 5003 it takes 2.5 to 7 times as long.

The reductions that compute the counts are not affected. Neither was a toy elementwise kernel we tried first, so how hard a kernel is hit depends on what the compiler makes of it. An M1 Pro (8 threads) agrees with the table. On an RTX 5060 Ti the step time is linear in the number of trees from 1 to 24, with no slow sizes.

### Effect on a fixed forest

This behaviour predates the variable number of trees. Time of a whole step on an M1 Pro (8 threads), 10 predictors, minimum over 40 interleaved repetitions:

| n | 4 trees, n even (ms) | 4 trees, n odd (ms) | Slowdown |
|---|---|---|---|
| 1 000 | 0.24 | 0.22 | none |
| 3 000 | 0.25 | 0.24 | none |
| 10 000 | 0.27 | 0.31 | 15% |
| 30 000 | 0.39 | 0.51 | 29% |
| 100 000 | 0.74 | 1.04 | 41% |
| 300 000 | 1.60 | 2.47 | 55% |
| 1 000 000 | 4.53 | 6.99 | 54% |

At n = 10⁶, 6 trees take 10.7 ms and 8 trees 8.6 ms. With 8 trees the parity of `n` makes no difference.

Padding the forest with trees not in use fixes it, provided the kernels then cover at least as many trees as threads. On the M1 Pro at n = 2·10⁶ a fixed forest of 5 trees takes 17.7 ms per step, and 8 trees with 5 in use 13.1 to 13.7 ms. Time of a whole step on the Ryzen 7 3800X (16 threads), 10 predictors, in milliseconds:

| Forest | n = 10 007 | n = 100 003 | n = 1 000 003 |
|---|---|---|---|
| 12 trees | 1.10 | 5.07 | 63.1 |
| 16 trees, 12 in use | 0.96 | 3.81 | 45.4 |
| 16 trees | 1.06 | 4.48 | 64.4 |
| 5 trees | | | 21.6 |
| 8 trees, 5 in use | | | 30.2 |
| 16 trees, 5 in use | | | 20.4 |

So 12 trees take as long as 16, and padding 5 trees to 8, still fewer than the threads, makes things worse.

The same on the Ryzen 9 7945HX (32 threads):

| Forest | n = 1 361 | n = 3 001 | n = 10 007 | n = 100 003 | n = 1 000 003 |
|---|---|---|---|---|---|
| 24 trees | 0.22 | 0.39 | 0.68 | 3.73 | 30.2 |
| 32 trees, 24 in use | 0.22 | 0.30 | 0.56 | 2.78 | 20.1 |
| 32 trees | 0.24 | 0.34 | 0.64 | 2.86 | 26.1 |
| 8 trees | | | | | 11.0 |
| 32 trees, 8 in use | | | | | 8.7 |
| 2 chains of 8 trees | | | | | 24.7 |
| 2 chains of 16 trees, 8 in use | | | | | 15.7 |
| 2 chains of 16 trees | | | | | 26.9 |

The slowdown of the whole step starts between 1361 and 3001 datapoints, earlier than for the prune kernel alone.

`CheckNumTreesCallback` warns, on the first iteration of an MCMC that runs on CPU, when the trees times the chains are more than one and fewer than the threads, and there are at least 2¹¹ datapoints. It compares with the threads rather than with the chunks, so it also warns in the few cases between the two.

### Effect on blocks

A block is a call over as many trees as the block size, so a block shall have one tree, or at least as many trees as threads. Parallel stage alone on an M1 Pro, 40 trees all in use, n = 2·10⁶ (even):

| Block size | Leftover block | Time (ms) |
|---|---|---|
| All at once | | 3.7 |
| 1 | | 8.5 |
| 5 | | 45.9 |
| 8 | | 6.8 |
| 9 | 4 | 6.9 |
| 11 | 7 | 16.3 |
| 12 | 4 | 6.4 |
| 13 | 1 | 6.1 |
| 17 | 6 | 14.4 |
| 35 | 5 | 13.2 |

A leftover block of a slow size costs as much as a slow block size, which is one reason the blocks are now all equal.

## Measurements with all the workarounds

These were taken with the sort, the barrier, and the default block sizes of the time, which for these configurations were at least as large as the thread counts (25 trees for 200 trees and n = 10⁶, a single block for 50 trees). All used round `n`.

Step time relative to all the trees at once, by fraction of trees in use:

| Machine | Configuration | 100% | 50% | 25% | 10% |
|---|---|---|---|---|---|
| M1 Pro, 8 threads | n = 10⁶, 200 trees | 1.03 | 1.00 | 0.89 | 0.85 |
| | n = 4·10⁶, 200 trees | 1.04 | 0.99 | 0.97 | 0.78 |
| | n = 10⁶, 1000 trees | 1.04 | 1.04 | 0.93 | 0.81 |
| | n = 10⁶, 200 trees, weights | 1.01 | 0.94 | 0.80 | 0.56 |
| Ryzen 7 5800X | n = 10⁶, 200 trees | 1.08 | 0.98 | 0.91 | 0.68 |
| | n = 4·10⁶, 200 trees | 1.03 | 0.95 | 0.84 | 0.61 |
| | n = 10⁶, 1000 trees | 1.04 | 0.96 | 0.79 | 0.59 |
| | n = 10⁶, 200 trees, weights | 1.11 | 0.99 | 0.79 | 0.56 |
| Xeon E5-2680 v4, 12 threads | n = 10⁶, 200 trees | 1.07 | 1.05 | 0.97 | 0.89 |
| | n = 4·10⁶, 200 trees | 1.08 | 1.05 | 1.03 | 0.84 |
| | n = 10⁶, 1000 trees | 1.08 | 1.06 | 0.99 | 0.85 |
| | n = 10⁶, 200 trees, weights | 1.11 | 1.05 | 0.94 | 0.78 |

Smaller problems (n ≤ 10⁵ with 200 trees) made a single block and were unchanged. The compilation of the step took roughly 10% longer.

Parallel stage alone on the M1 Pro, in milliseconds, by number of trees in use:

| Configuration | Variant | 100% | 50% | 25% | 10% |
|---|---|---|---|---|---|
| n = 10⁶, 200 trees | All at once | 9.4 | 8.4 | 8.3 | 8.3 |
| | Blocks of 25 | 15.5 | 7.6 | 4.0 | 2.2 |
| | Blocks of 1 | 24.8 | 12.6 | 6.6 | 2.9 |
| n = 10⁵, 1000 trees | All at once | 5.7 | 5.1 | 5.0 | 5.0 |
| | Blocks of 125 | 9.3 | 5.1 | 2.9 | 1.8 |
| | Blocks of 1 | 34.7 | 17.7 | 9.4 | 3.9 |
| n = 2·10⁶, 40 trees | All at once | 3.7 | 3.7 | 3.5 | 3.5 |
| | Blocks of 5 (slow size) | 45.9 | 23.0 | 11.6 | 5.9 |
| | Blocks of 1 | 8.5 | 4.3 | 2.3 | 1.1 |

Over 14 sizes from 7·10⁷ to 10⁹ (tree, datapoint) pairs, with all the trees in use the blocks took 1.6 to 2.5 times as long as all the trees at once, a few of them with a slow block size. Blocks of a good size broke even between 50% and 60% in use. Blocks of one tree broke even between 15% in use (n = 10⁵) and 45% (n ≥ 2·10⁶).

### Switching at half in use

Since blocks lose above half in use, we tried choosing at run time:

```python
def run_cpu():
    return lax.cond(
        2 * vmap_max(forest.used_trees) > num_trees,
        run,  # all the trees at once
        partial(run, block_size, materialize=True),
    )
```

It did what it should: on the M1 Pro and the Xeon the step with all the trees in use went back to 0.99 to 1.01 of the time without blocks, with the gains at 25% and 10% in use unchanged, and no copy of the leaf indices. Compiling both branches added 0 to 0.18 s to the compilation of the step, independently of the sizes. With all the trees in use it saved 4 to 37 ms per step on the M1 Pro, so it paid for itself in tens of steps for any size at which blocks are used at all.

## What the code does now

- On CPU, `update_leaf_indices_and_related_caches` processes all the trees at once. `StepConfig.parallel_block_size` still forces blocks on any backend, to test them and to experiment, without the workarounds of sections 1 and 2.
- The blocks are all equal: the block size divides the number of trees, and `init` asks for a multiple of 8 trees when `used_trees` is set so that a suitable divisor always exists.
- `CheckNumTreesCallback` warns at run time on CPU about a forest with fewer trees (times chains) than threads. `init` can not, because the state may move to another platform before the MCMC runs. `Bart` does not use the callback yet.

So on CPU a step costs the same whatever the fraction of trees in use, in the part vectorized over trees.

## What is left to try

1. **Blocks of at least as many trees as threads, with the switch at half in use.** This is the configuration measured above, with the block size tied to the thread count. It needs the sort, the barrier and the conditional back. Expected gain: 10% to 45% of the step with a tenth of the trees in use, nothing above half. On a machine with many threads the blocks become coarse.
2. **Blocks of one tree.** They need no knowledge of the thread count, and win only below 15% to 45% in use.
3. **Not writing into a slice.** The serial copy of section 2 comes from keeping the leaf indices in one array. Keeping them as separate arrays per block would remove it, at the cost of changing how the forest is stored. Not tried.
4. **A fixed forest smaller than the thread count.** Updating its trees one at a time, or padding the kernels to the thread count internally, would remove the need for the warning. Not tried.
5. **Reporting sections 2 and 3 to XLA.** Both may be missed optimizations rather than design decisions.

## Reproducing

This prints the partitions XLA picks for an elementwise update with a per-tree operand, and times it. Run it with different `T`, and under `taskset -c 0-<P-1>` to change the number of threads.

```python
import re
import sys
import time

import jax
from jax import numpy as jnp

T, n = int(sys.argv[1]), int(sys.argv[2])


@jax.jit
def prune(leaf_indices, to_prune, move_node):
    mask = ~jnp.array(1, leaf_indices.dtype)
    is_child = (leaf_indices & mask) == (move_node[:, None] << 1)
    node = move_node[:, None].astype(leaf_indices.dtype)
    return jnp.where(is_child & to_prune[:, None], node, leaf_indices)


args = jnp.ones((T, n), jnp.uint8), jnp.ones(T, bool), jnp.ones(T, jnp.int32)
text = prune.lower(*args).compile().as_text()
print(re.findall(r'outer_dimension_partitions[^}\]]*[\]}]', text))
prune(*args).block_until_ready()
times = []
for _ in range(30):
    start = time.perf_counter()
    prune(*args).block_until_ready()
    times.append(time.perf_counter() - start)
print(f'{1e6 * min(times) / T:.1f} us per tree')
```

Setting `TF_CPP_MIN_LOG_LEVEL=0` and `TF_CPP_VMODULE=<xla source file>=<level>` in the environment makes jaxlib print the logs of an XLA pass.
