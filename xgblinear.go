package leaves

// xgLinear is XGBoost model (gblinear)
type xgLinear struct {
	NumFeature int
	nClasses   int
	BaseScore  float64
	Weights    []float32
}

func (e *xgLinear) NEstimators() int {
	return 1
}

func (e *xgLinear) NClasses() int {
	return e.nClasses
}

func (e *xgLinear) adjustNEstimators(nEstimators int) int {
	// gbliearn has only one estimator per class
	return 1
}

func (e *xgLinear) NFeatures() int {
	return e.NumFeature
}

func (e *xgLinear) Name() string {
	return "xgboost.gblinear"
}

func (e *xgLinear) predictInner(fvals []float64, nIterations int, predictions []float64, startIndex int) {
	for k := 0; k < e.nClasses; k++ {
		predictions[startIndex+k] = e.BaseScore + float64(e.Weights[e.nClasses*e.NumFeature+k])
		for i := 0; i < e.NumFeature; i++ {
			predictions[startIndex+k] += fvals[i] * float64(e.Weights[e.nClasses*i+k])
		}
	}
}

func (e *xgLinear) predictInnerSparse(fvals map[uint32]float64, nEstimators int, predictions []float64, startIndex int) {
	for k := 0; k < e.nClasses; k++ {
		predictions[startIndex+k] = e.BaseScore + float64(e.Weights[e.nClasses*e.NumFeature+k])
		for i := 0; i < e.NumFeature; i++ {
			predictions[startIndex+k] += fvals[uint32(i)] * float64(e.Weights[e.nClasses*i+k])
		}
	}
}

func (e *xgLinear) predictInnerLeafSparse(fvals map[uint32]float64, nEstimators int, ret []int) {
	predictions := make([]float64, e.nClasses)
	e.predictInnerSparse(fvals, nEstimators, predictions, 0)

	maxIndex := 0
	maxVal := predictions[0]
	for i := 1; i < e.nClasses; i++ {
		if predictions[i] > maxVal {
			maxIndex = i
			maxVal = predictions[i]
		}
	}
	ret[0] = maxIndex
}

func (e *xgLinear) getLeafSize(nEstimators int) int {
	return e.nClasses
}

func (e *xgLinear) resetFVals(fvals []float64) {
	for j := 0; j < len(fvals); j++ {
		fvals[j] = 0.0
	}
}
