package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"gonum.org/v1/gonum/stat/combin"

	"github.com/btracey/btutil"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

func main() {
	for _, dataset := range []string{"turchinFirst_onerep", "turchinFirst", "turchinSecond"} {
		//for _, dataset := range []string{"turchinFirst"} {

		var path, name string
		var loadCols map[int]int
		loadReps := make(map[int]struct{})
		repIdx := -1
		switch dataset {
		default:
			panic("unknown")
		case "turchinFirst", "turchinFirst_onerep":
			// Load in the imputed data.
			path = filepath.Join("/", "Users", "brendan", "Dropbox (MIT)", "turchin")
			name = "imputed.csv"
			loadCols = make(map[int]int)
			for i := 0; i < 9; i++ {
				loadCols[i+3] = i
			}
			repIdx = 12
			switch dataset {
			default:
				panic("unknown")
			case "turchinFirst":
				loadReps[-1] = struct{}{}
			case "turchinFirst_onerep":
				loadReps[1] = struct{}{} // rep numbers one indexed
			}
		case "turchinSecond":
			// Load in the results from running Turchin's code.
			path = filepath.Join("/", "Users", "brendan", "Documents", "SFI", "codeother", "turchin", "turchin2run")
			name = "ImpDatRepl.csv"
			loadCols = map[int]int{4: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
			loadReps[-1] = struct{}{}
		}

		prefix := dataset + "_"
		// Load in the file and read the CSV
		filename := filepath.Join(path, name)
		f, err := os.Open(filename)
		if err != nil {
			log.Fatal(err)
		}
		r := csv.NewReader(f)
		csvdata, err := r.ReadAll()
		if err != nil {
			log.Fatal(err)
		}

		// Just get the subset of the CSV data.
		var nIncludedCols int
		dataNonNorm := mat.NewDense(len(csvdata)-1, len(loadCols), nil)
		// Start from one because of the header column.
		for i := 1; i < len(csvdata); i++ {
			// Skip repetitions if we only want a subset.
			// Negative one means load all of the repititions
			if _, ok := loadReps[-1]; !ok {
				repNumStr := csvdata[i][repIdx]
				repNum, err := strconv.Atoi(repNumStr)
				if err != nil {
					log.Fatal(err)
				}
				fmt.Println(repNum)
				// If in map, then we should load
				if _, ok := loadReps[repNum]; !ok {
					continue
				}
			}

			// Load in the relevant columns.
			for j := range csvdata[0] {
				if _, ok := loadCols[j]; !ok {
					continue
				}
				str := csvdata[i][j]
				v, err := strconv.ParseFloat(str, 64)
				if err != nil {
					log.Fatal(err)
				}
				dataNonNorm.Set(nIncludedCols, loadCols[j], v)
			}
			nIncludedCols++
		}
		fmt.Println(nIncludedCols, len(csvdata))
		dataNonNorm = dataNonNorm.Slice(0, nIncludedCols, 0, len(loadCols)).(*mat.Dense)

		// Transform the matrix to be normalized.
		m, n := dataNonNorm.Dims()
		data := mat.NewDense(m, n, nil)
		for j := 0; j < n; j++ {
			col := mat.Col(nil, j, dataNonNorm)
			mean := stat.Mean(col, nil)
			std := stat.StdDev(col, nil)
			fmt.Println("mean", mean, "std", std)
			for i := 0; i < m; i++ {
				v := dataNonNorm.At(i, j)
				v = (v - mean) / std
				data.Set(i, j, v)
			}
		}
		covData := stat.CovarianceMatrix(nil, data, nil)
		btutil.PrintMat("covdata", covData)

		// Print the historgram of all of the CC's
		for i := 0; i < n; i++ {
			makeHistogramColumn(data, i, "rawcol/"+prefix+"rawcol_"+strconv.Itoa(i)+".pdf")
		}

		// Compute PCA of the data.
		pca := &stat.PC{}
		pca.PrincipalComponents(data, nil)
		values := pca.VarsTo(nil)
		vectors := pca.VectorsTo(nil)
		sumValues := floats.Sum(values)
		fmt.Println(values)
		fmt.Println("Percent first", values[0]/sumValues)
		firstVector := mat.Col(nil, 0, vectors)
		fmt.Println("firstVector", firstVector)

		// Project the points to the first principal component and plot histogram.
		proj := &mat.Dense{}
		proj.Mul(data, vectors)

		// Double check this was done right. Off diagonal elements should be 0.
		cov := stat.CovarianceMatrix(nil, proj, nil)
		btutil.PrintMat("cov", cov)

		// Make histogram of first principal component.
		makeFirstHist(proj, prefix)

		// Make scatter plot of the first two components.
		makeScatFirstTwo(proj, prefix+"pcascat.pdf")

		// Make plots where we zero out the various CC's and then plot the
		// points projected onto the first two principal components.
		// This code here is indexing over all the combinations of CC's being
		// kept in or removed.
		maxIdx := 1
		dims := make([]int, n)
		for i := 0; i < n; i++ {
			maxIdx *= 2
			dims[i] = 2
		}
		sub := make([]int, n)
		dataZero := &mat.Dense{}
		projZero := &mat.Dense{}
		for idx := 0; idx < maxIdx; idx++ {
			// One means erase
			combin.SubFor(sub, idx, dims)
			var subSum int
			for _, v := range sub {
				subSum += v
			}
			// If erasing a medium number of CC's, skip, so we don't generate
			// a bazillion plots.
			if subSum > 3 && subSum < n-2 {
				continue
			}
			dataZero.Clone(data)
			var notMissing string
			var numMissing int

			for j := 0; j < n; j++ {
				if sub[j] == 0 {
					notMissing += "_" + strconv.Itoa(j)
					continue
				}
				numMissing++
				for i := 0; i < m; i++ {
					dataZero.Set(i, j, 0)
				}
			}
			projZero.Mul(dataZero, vectors)
			makeScatFirstTwo(projZero, "ccremove/"+prefix+"pcascat_zero"+"_"+strconv.Itoa(numMissing)+notMissing+".pdf")
		}
	}
}

func makeHistogramColumn(mat *mat.Dense, col int, name string) {
	m, _ := mat.Dims()
	pts := make(plotter.Values, m)
	for i := range pts {
		pts[i] = mat.At(i, col)
	}
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	h, err := plotter.NewHist(pts, 400)
	if err != nil {
		panic(err)
	}
	p.Add(h)
	p.Save(4.5*vg.Inch, 3*vg.Inch, name)
}

func makeFirstHist(proj *mat.Dense, prefix string) {
	m, _ := proj.Dims()
	pts := make(plotter.Values, m)
	for i := range pts {
		pts[i] = proj.At(i, 0)
	}
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	h, err := plotter.NewHist(pts, 400)
	if err != nil {
		panic(err)
	}
	p.Add(h)
	p.Save(4.5*vg.Inch, 3*vg.Inch, prefix+"pcahist.pdf")
}

func makeScatFirstTwo(proj *mat.Dense, name string) {
	m, _ := proj.Dims()
	pts := make(plotter.XYs, m)
	for i := range pts {
		pts[i].X = proj.At(i, 0)
		pts[i].Y = proj.At(i, 1)
	}
	scat, err := plotter.NewScatter(pts)
	scat.Shape = draw.PyramidGlyph{}
	scat.Radius = 1
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.Add(scat)
	p.Save(5*vg.Inch, 3*vg.Inch, name)
}
