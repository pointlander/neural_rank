// Copyright 2020 The Neural Rank Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand"
	"sort"

	"github.com/pointlander/gradient/tc128"
	"github.com/pointlander/gradient/tf32"
	"github.com/pointlander/pagerank"

	"github.com/texttheater/golang-levenshtein/levenshtein"
)

const (
	// Size is the size of the adjacency matrix
	Size = 5
)

// Ranks retunrs an array of ranks
func Ranks(ranks []float32) []rune {
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
	}
	return sorted
}

// RankCompare compares two ranking function
func RankCompare(set *tf32.Set) []int {
	distances := make([]int, 2)

	a := set.ByName["A"]

	x := tf32.NewV(Size)
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
		if total < 1e-6 {
			break
		}
	}
	nonlinear := Ranks(x.X)

	var nonlinearMiddle []rune
	l1(func(a *tf32.V) bool {
		nonlinearMiddle = Ranks(a.X)
		return true
	})

	graph, index := pagerank.NewGraph64(), 0
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			x := a.X[index]
			if x != 0 {
				graph.Link(uint64(j), uint64(i), float64(x))
			}
			index++
		}
	}
	ranks := make([]float32, Size)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = float32(rank)
	})
	linear := Ranks(ranks)

	distances[0] = levenshtein.DistanceForStrings(nonlinearMiddle, linear, levenshtein.DefaultOptions)
	distances[1] = levenshtein.DistanceForStrings(nonlinear, linear, levenshtein.DefaultOptions)
	return distances
}

// RanksComplex retunrs an array of ranks
func RanksComplex(ranks []complex128) []rune {
	sorted := make([]rune, 0, 8)
	type Pair struct {
		Index int
		Rank  float32
	}
	pairs := make([]Pair, len(ranks))
	for i := range pairs {
		pairs[i].Index = i
		pairs[i].Rank = float32(cmplx.Abs(ranks[i]))
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Rank < pairs[j].Rank
	})
	for i := range pairs {
		sorted = append(sorted, rune(pairs[i].Index))
	}
	return sorted
}

// RankCompareComplex compares two ranking function
func RankCompareComplex(set *tf32.Set) []int {
	distances := make([]int, 2)
	a := set.ByName["A"]

	s := tc128.NewSet()
	s.Add("A", Size, Size)
	ac := s.ByName["A"]

	index := 0
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			forward, back := 0.0, float64(a.X[index])
			for k := 0; k < Size; k++ {
				forward += float64(a.X[k*Size+j])
			}
			// https://www.sciencedirect.com/science/article/pii/S0972860019300945
			ac.X = append(ac.X, complex((forward+back)/2, (forward-back)/2))
			index++
		}
	}

	x := tc128.NewV(Size)
	x.X = x.X[:cap(x.X)]

	deltas := make([]complex128, len(x.X))

	l1 := tc128.Softmax(tc128.Mul(s.Get("A"), x.Meta()))
	l2 := tc128.Softmax(tc128.Mul(s.Get("A"), l1))
	cost := tc128.Avg(tc128.Quadratic(x.Meta(), l2))

	iterations := 8
	alpha, eta := complex(.3, 0), complex(.3, 0)
	for i := 0; i < iterations; i++ {
		s.Zero()
		x.Zero()

		total := tc128.Gradient(cost).X[0]
		norm := complex128(0)
		for _, d := range x.D {
			norm += d * d
		}
		norm = cmplx.Sqrt(norm)
		scaling := complex(1, 0)
		if cmplx.Abs(norm) > 1 {
			scaling = 1 / norm
		}
		for l, d := range x.D {
			deltas[l] = alpha*deltas[l] - eta*d*scaling
			x.X[l] += deltas[l]
		}
		if cmplx.Abs(total) < 1e-6 {
			break
		}
	}
	nonlinear := RanksComplex(x.X)

	var nonlinearMiddle []rune
	l1(func(a *tc128.V) bool {
		nonlinearMiddle = RanksComplex(a.X)
		return true
	})

	graph, index := pagerank.NewGraph64(), 0
	for i := 0; i < Size; i++ {
		for j := 0; j < Size; j++ {
			x := a.X[index]
			if x != 0 {
				graph.Link(uint64(j), uint64(i), float64(x))
			}
			index++
		}
	}
	ranks := make([]float32, Size)
	graph.Rank(0.85, 0.000001, func(node uint64, rank float64) {
		ranks[node] = float32(rank)
	})
	linear := Ranks(ranks)

	distances[0] = levenshtein.DistanceForStrings(nonlinearMiddle, linear, levenshtein.DefaultOptions)
	distances[1] = levenshtein.DistanceForStrings(nonlinear, linear, levenshtein.DefaultOptions)

	return distances
}

func main() {
	rand.Seed(1)

	set := tf32.NewSet()
	set.Add("A", Size, Size)
	a := set.ByName["A"]
	a.X = append(a.X, 0, 0, 0, 0, 1)
	a.X = append(a.X, .5, 0, 0, 0, 0)
	a.X = append(a.X, .5, 0, 0, 0, 0)
	a.X = append(a.X, 0, 1, .5, 0, 0)
	a.X = append(a.X, 0, 0, .5, 1, 0)
	distances := RankCompare(&set)
	fmt.Println(distances)

	distances = RankCompareComplex(&set)
	fmt.Println(distances)

	distances, distancesComplex := make([]int, 2), make([]int, 2)
	for i := 0; i < 1024; i++ {
		for i := range a.X {
			if rand.Intn(2) == 0 {
				a.X[i] = float32(math.Abs(rand.NormFloat64()))
			}
		}
		d := RankCompare(&set)
		for i, v := range d {
			distances[i] += v
		}
		d = RankCompareComplex(&set)
		for i, v := range d {
			distancesComplex[i] += v
		}
	}
	fmt.Println(float64(distances[0])/1024, float64(distances[1])/1024)
	fmt.Println(float64(distancesComplex[0])/1024, float64(distancesComplex[1])/1024)
}
