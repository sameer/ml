package id3

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sort"
)

type Decision struct {
	nextDecisions map[Feature]*Decision
	featureName   string
	isOutput      bool
	outputValue   Target
}

func (dtree *Decision) String() []string {
	paths := dtree.string(nil)
	sort.Strings(paths)
	return paths
}

func (dtree *Decision) string(parents []*Decision) []string {
	if dtree.isOutput {
		sout := ""
		for i, parent := range parents {
			var featureVal Feature
			var pChild *Decision
			if i+1 < len(parents) {
				pChild = parents[i+1]
			} else {
				pChild = dtree
			}
			for k, v := range parent.nextDecisions {
				if v == pChild {
					featureVal = k
					break
				}
			}
			sout += fmt.Sprintf("%v[%v] ==> ", parent.featureName, featureVal)
		}
		sout += fmt.Sprintf("%#v", dtree.outputValue)
		return []string{sout}
	} else {
		sout := []string{}
		if parents == nil {
			parents = make([]*Decision, 0)
		}
		parents = append(parents, dtree)
		for _, subtree := range dtree.nextDecisions {
			sout = append(sout, subtree.string(parents)...)
		}
		return sout
	}
}

type Feature uint8

type Target bool

type ClassifiedDataSet struct {
	Instances []*Instance
}

type Instance struct {
	FeatureValues map[string]Feature
	TargetValue   Target
}

type BestFeatureFunc func(ds ClassifiedDataSet) string

func Train(ds ClassifiedDataSet, bf BestFeatureFunc) (*Decision, error) {
	dtree := &Decision{}
	if ds.Instances == nil || len(ds.Instances) == 0 {
		return nil, errors.New("No instances provided")
	} else if dtree.featureName = bf(ds); dtree.featureName == "" { // No features left
		dtree.outputValue, dtree.isOutput = mostPopularTarget(ds.Instances), true
		return dtree, nil
	} else if instancesIdentical(ds.Instances) {
		dtree.outputValue, dtree.isOutput = ds.Instances[0].TargetValue, true
		return dtree, nil
	} else {
		dtree.nextDecisions = make(map[Feature]*Decision)
		bestFeatureValToInstances := make(map[Feature][]*Instance)
		for _, inst := range ds.Instances {
			instances, ok := bestFeatureValToInstances[inst.FeatureValues[dtree.featureName]]
			if !ok {
				instances = make([]*Instance, 0)
			}
			bestFeatureValToInstances[inst.FeatureValues[dtree.featureName]] = append(instances, inst)
		}
		for _, inst := range ds.Instances {
			delete(inst.FeatureValues, dtree.featureName)
		}
		for k, v := range bestFeatureValToInstances {
			var err error
			dtree.nextDecisions[k], err = Train(ClassifiedDataSet{Instances: v}, bf)
			if err != nil {
				return nil, errors.New(fmt.Sprint("No instances available to extend tree for feature", dtree.featureName, "with value", k, "this shouldn't be possible"))
			}
		}
		return dtree, nil
	}
}

func CalculateError(dtree *Decision, ds ClassifiedDataSet) (float64, error) {
	var wrongClassifications float64 = 0.0
	for _, inst := range ds.Instances {
		correctTargetValue := inst.TargetValue
		if err := Classify(dtree, inst); err != nil {
			return 1.0, err
		} else if correctTargetValue != inst.TargetValue {
			wrongClassifications++
		}
		inst.TargetValue = correctTargetValue
	}
	return wrongClassifications / float64(len(ds.Instances)), nil
}

func Classify(dtree *Decision, inst *Instance) error {
	if dtree.isOutput {
		inst.TargetValue = dtree.outputValue
		return nil
	} else if thisValue, ok := inst.FeatureValues[dtree.featureName]; ok {
		if nextDecision, ok := dtree.nextDecisions[thisValue]; ok {
			return Classify(nextDecision, inst)
		} else {
			return errors.New(fmt.Sprint("No decision node corresponding to instance value of", thisValue, "for", dtree.featureName))
		}
	} else {
		return errors.New(fmt.Sprint("No decision node for feature", dtree.featureName))
	}
}

func instancesIdentical(insts []*Instance) bool {
	for i := 1; i < len(insts); i++ {
		if !reflect.DeepEqual(*insts[i], *insts[i-1]) {
			return false
		}
	}
	return true
}

func mostPopularTarget(insts []*Instance) Target {
	targetCounts := make(map[Target]int)
	highestCount := 0
	var highestTarget Target
	for _, inst := range insts {
		count, ok := targetCounts[inst.TargetValue]
		if !ok {
			count = 0
		}
		count++
		targetCounts[inst.TargetValue] = count
		if count > highestCount {
			highestCount = count
			highestTarget = inst.TargetValue
		}
	}
	return highestTarget
}

func BestFeatureInformationGain(ds ClassifiedDataSet) string {
	var greatestInfoGain float64 = 0.0
	greatestFeature := ""
	for featureName := range ds.Instances[0].FeatureValues {
		infoGain := infoGainOfFeature(ds, featureName)
		if infoGain > greatestInfoGain {
			greatestInfoGain = infoGain
			greatestFeature = featureName
		}
	}
	return greatestFeature
}

var _ BestFeatureFunc = BestFeatureInformationGain

func infoGainOfFeature(ds ClassifiedDataSet, featureName string) float64 {
	featureValueCounts := make(map[Feature]int)
	for _, inst := range ds.Instances {
		count, ok := featureValueCounts[inst.FeatureValues[featureName]]
		if !ok {
			count = 0
		}
		count++
		featureValueCounts[inst.FeatureValues[featureName]] = count
	}

	var infoGain float64 = entropy(ds.Instances)
	for featureValue, featureCount := range featureValueCounts {
		featureValueEntropy := entropy(filter(ds.Instances, func(inst *Instance) bool {
			return inst.FeatureValues[featureName] == featureValue
		}))
		infoGain -= float64(featureCount) / float64(len(ds.Instances)) * featureValueEntropy
	}

	return infoGain
}

func filter(insts []*Instance, keep func(inst *Instance) bool) []*Instance {
	retInsts := make([]*Instance, 0)
	for _, inst := range insts {
		if keep(inst) {
			retInsts = append(retInsts, inst)
		}
	}
	return retInsts
}

func entropy(insts []*Instance) float64 {
	targetCounts := make(map[Target]int)
	for _, inst := range insts {
		count, ok := targetCounts[inst.TargetValue]
		if !ok {
			count = 0
		}
		count++
		targetCounts[inst.TargetValue] = count
	}
	var H float64 = 0.0
	for _, count := range targetCounts {
		pI := float64(count) / float64(len(insts))
		H += pI * math.Log2(pI)
	}
	return -H
}
