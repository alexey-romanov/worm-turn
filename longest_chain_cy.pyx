# C. elegans worm turn detector

# MIT License

# Copyright (c) 2023 Aleksei Romanov

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# cython: language_level=3
# distutils: language = c++

import numpy as np

cimport numpy as np
cimport cython

np.import_array()

from libcpp.deque cimport deque
from libcpp.vector cimport vector

cdef struct Node:
    char visited
    int prev_x
    int prev_y

cdef int is_valid(vector[vector[Node]]& graph, int idx_x, int idx_y, int shape_x, int shape_y):
    if idx_x < 0:
        return 0
    if idx_y < 0:
        return 0
    if idx_x >= shape_x:
        return 0
    if idx_y >= shape_y:
        return 0

    if graph[idx_x][idx_y].visited == 1:
        return 0

    return 1


def longest_chain(np.ndarray img, int start_idx_x, int start_idx_y, int min_idx_x, int min_idx_y):
    chain = []

    cdef vector[vector[Node]] graph

    cdef int W = img.shape[0]
    cdef int H = img.shape[1]

    graph.resize(W)
    for i in range(W):
        graph[i].resize(H)

    cdef int xI
    cdef int yI
    for xI from 0 <= xI < W by 1:
        for yI from 0 <= yI < H by 1:
            graph[xI][yI].prev_x = -1
            graph[xI][yI].prev_y = -1

            if img[(xI, yI)] == 0:
                graph[xI][yI].visited = 1
            else:
                graph[xI][yI].visited = 0


    cdef deque[int] iter_set

    iter_set.push_back(start_idx_x - min_idx_x)
    iter_set.push_back(start_idx_y - min_idx_y)

    cdef int idx_test_x
    cdef int idx_test_y

    cdef int xl
    cdef int yl

    cdef int idx0
    cdef int idx1

    cdef int backtrace_idx_0
    cdef int backtrace_idx_1

    cdef int temp0
    cdef int temp1

    while iter_set.size() != 0:
        idx0 = iter_set.front();
        iter_set.pop_front()
        idx1 = iter_set.front();
        iter_set.pop_front()

        graph[idx0][idx1].visited = 1

        for xl from 0 <= xl < 3 by 1:
            for yl from 0 <= yl < 3 by 1:
                if xl == 1 and yl == 1:
                    continue
                idx_test_x = idx0 + xl - 1
                idx_test_y = idx1 + yl - 1
                if is_valid(graph, idx_test_x, idx_test_y, img.shape[0], img.shape[1]) == 1:

                    iter_set.push_back(idx_test_x)
                    iter_set.push_back(idx_test_y)

                    graph[idx_test_x][idx_test_y].prev_x = idx0
                    graph[idx_test_x][idx_test_y].prev_y = idx1

        if iter_set.size() == 0:
            backtrace_idx_0 = idx0
            backtrace_idx_1 = idx1
            while backtrace_idx_0 != -1 and backtrace_idx_1 != -1:
                chain.append((backtrace_idx_0 + min_idx_x, backtrace_idx_1 + min_idx_y))

                temp0 = backtrace_idx_0
                temp1 = backtrace_idx_1

                backtrace_idx_0 = graph[temp0][temp1].prev_x
                backtrace_idx_1 = graph[temp0][temp1].prev_y

    return chain
