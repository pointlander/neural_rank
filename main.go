// Copyright 2020 The Neural Rank Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"sort"

	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/pagerank"

	"github.com/texttheater/golang-levenshtein/levenshtein"
)

// PrintRanks prints the ranks
func PrintRanks(ranks []float32) []rune {
	sorted := make([]rune, 0, 8)
	type Pair struct {
		Index int
		Rank  float32
	}
	pairs := make([]Pair, len(ranks))
	for i := range pairs {
		pairs[i].Index = i
		pairs[i].Rank = ranks[i]
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Rank < pairs[j].Rank
	})
	for i := range pairs {
		sorted = append(sorted, rune(pairs[i].Index))
		fmt.Printf("%d ", pairs[i].Index)
	}
	fmt.Printf("\n")
	return sorted
}

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

	l1 := tf32.Softmax(tf32.Mul(set.Get("A"), x.Meta()))
	l2 := tf32.Softmax(tf32.Mul(set.Get("A"), l1))
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
	nonlinear := PrintRanks(x.X)

	l1(func(a *tf32.V) bool {
		PrintRanks(a.X)
		return true
	})

	graph := pagerank.NewGraph64()
	graph.Link(4, 0, 1)
	graph.Link(0, 1, .5)
	graph.Link(0, 2, .5)
	graph.Link(1, 3, 1)
	graph.Link(2, 3, .5)
	graph.Link(2, 4, .5)
	graph.Link(3, 4, 1)
	ranks := make([]float32, 5)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = float32(rank)
	})
	linear := PrintRanks(ranks)

	distance := levenshtein.DistanceForStrings(nonlinear, linear, levenshtein.DefaultOptions)
	fmt.Println(distance)
}
