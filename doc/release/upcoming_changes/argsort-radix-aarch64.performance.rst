``np.argsort`` is faster for integer keys on non-x86 (aarch64, etc.)
--------------------------------------------------------------------
On platforms without a SIMD argsort path (non-x86, e.g. aarch64/ppc64),
``numpy.argsort`` (default ``kind='quicksort'``) now routes large (>= 16384)
*disordered* integer arrays to the existing scalar radix argsort instead of the
scalar comparison sort -- a large win on random, low-cardinality and
block-structured data. Sorted, near-sorted and reverse input stay on the
comparison sort (gated by a cheap descent + block-mean-trend check), so they are
not regressed. x86, floating point and descending argsort are unchanged.

Because the radix argsort is stable, the order of equal keys for
``kind='quicksort'`` may differ from x86 (where the SIMD quicksort is unstable);
that kind does not guarantee stability.

Measured with NumPy's ``bench_function_base.Sort`` generators through
``np.argsort`` on Apple M1 Pro (NEON), 1e6 elements, baseline -> radix:

* ``int64``  ``random`` 68 -> 12 ns/key (5.5x); ``sorted_block`` 3.5-4.1x
* ``int32``  ``random`` 67 -> 10 ns/key (6.5x)
* ``int16``  ``random`` 66 ->  9 ns/key (7.4x)

Cross-checked on AWS Graviton3.
