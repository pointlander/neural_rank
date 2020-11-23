// Copyright 2020 The Neural Rank Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/pagerank"
)

func main() {
	set := tf32.NewSet()
	set.Add("A", 5, 5)

	a := set.ByName["A"]
	a.X = append(a.X, 0, 0, 0, 0, 1)
	a.X = append(a.X, .5, 0, 0, 0, 0)
	a.X = append(a.X, .5, 0, 0, 0, 0)
	a.X = append(a.X, 0, 1, .5, 0, 0)
	a.X = append(a.X, 0, 0, .5, 1, 0)

	x := tf32.NewV(5)
	x.X = x.X[:cap(x.X)]

	deltas := make([]float32, len(x.X))

	l1 := tf32.Sigmoid(tf32.Mul(set.Get("A"), x.Meta()))
	l2 := tf32.Sigmoid(tf32.Mul(set.Get("A"), l1))
	cost := tf32.Avg(tf32.Quadratic(x.Meta(), l2))

	iterations := 8
	alpha, eta := float32(.3), float32(.3)
	for i := 0; i < iterations; i++ {
		set.Zero()
		x.Zero()

		total := tf32.Gradient(cost).X[0]
		norm := float32(0)
		for _, d := range x.D {
			norm += d * d
		}
		norm = float32(math.Sqrt(float64(norm)))
		scaling := float32(1)
		if norm > 1 {
			scaling = 1 / norm
		}
		for l, d := range x.D {
			deltas[l] = alpha*deltas[l] - eta*d*scaling
			x.X[l] += deltas[l]
		}
		fmt.Println(total)
	}
	fmt.Println(x.X)

	l1(func(a *tf32.V) bool {
		fmt.Println(a.X)
		return true
	})

	graph := pagerank.NewGraph64()
	graph.Link(0, 4, 1)
	graph.Link(1, 0, .5)
	graph.Link(2, 0, .5)
	graph.Link(3, 1, 1)
	graph.Link(3, 2, .5)
	graph.Link(4, 2, .5)
	graph.Link(4, 3, 1)
	ranks := make([]float32, 5)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = float32(rank)
	})
	fmt.Println(ranks)
}
