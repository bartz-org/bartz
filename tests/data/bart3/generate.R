# bartz/tests/data/bart3/generate.R
#
# Copyright (c) 2026, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Generate the R BART3 tree trace used to test `bartz.debug.trees_BART_to_bartz`.
# Requires R with BART3 (https://github.com/rsparapa/bnptools) installed. Run
# from this directory with `Rscript generate.R`.

library(BART3)

set.seed(20261016)
p <- 3
n <- 30
x <- matrix(rnorm(p * n), p, n)
y <- x[1, ] - 0.5 * x[2, ] * x[3, ] + rnorm(n, 0, 0.3)
xinfo <- matrix(rep(seq(-1.5, 1.5, length.out = 10), p), p, byrow = TRUE)

# with transposed = TRUE, `numcut` must be a vector, otherwise BART3 hangs
fit <- gbart(
    x.train = x, y.train = y, transposed = TRUE, xinfo = xinfo,
    numcut = rep(ncol(xinfo), p), ntree = 10, ndpost = 20, nskip = 100,
    rm.const = FALSE, printevery = 1000
)

write_matrix <- function(m, file) {
    write.table(m, file, row.names = FALSE, col.names = FALSE)
}
writeLines(fit$treedraws$trees, 'trees.txt', sep = '')
write_matrix(x, 'x_train.txt')
write_matrix(xinfo, 'xinfo.txt')
write_matrix(fit$offset, 'offset.txt')
write_matrix(fit$yhat.train, 'yhat_train.txt')
write_matrix(fit$varcount, 'varcount.txt')
