package id3

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

// Decision tree node type.
// If it is not an output node, it keeps track of the name of the feature
// being used and the child Decisions.
// if it is an output node, it keeps track of its output value.
type Decision struct {
	nextDecisions map[Feature]*Decision
	featureName   string
	isOutput      bool
	outputValue   Target
}

// Convert a decision tree to a sorted string slice of all possible paths to output nodes.
// Useful for debugging or equality-check purposes.
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

// The type used for decision tree features. Up to 256 discrete values are allowed.
// The trainer builds the tree assuming that the only possible feature values are those specified
// in the provided dataset
type Feature uint8

// The type used for decision tree targets, or outputs.
type Target bool

// A set of pointers to classified data.
type ClassifiedDataSet struct {
	Instances []*Instance
}

// A piece of data. It can be considered classified or unclassified. When used in a ClassifiedDataSet, it should
// always be classified.
type Instance struct {
	FeatureValues map[string]Feature
	TargetValue   Target
}

// Creates a duplicate or clone of an instance.
func (i *Instance) Clone() *Instance {
	clone := &Instance{}
	clone.TargetValue = i.TargetValue
	clone.FeatureValues = make(map[string]Feature)
	for k, v := range i.FeatureValues {
		clone.FeatureValues[k] = v
	}
	return clone
}

// A type of function that selects the best feature for the decision tree to build upon.
// One BestFeatureFunc using information gain is provided.
type BestFeatureFunc func(ds ClassifiedDataSet) string

// Using a classified set of data and the provided BestFeatureFunc, the ID3 algorithm is run to train and return
// a decision tree.
func Train(ds ClassifiedDataSet, bf BestFeatureFunc) (*Decision, error) {
	return BoundedTrain(ds, bf, ^uint(0))
}

// Allows for training with a specified maximum number of iterations
func BoundedTrain(ds ClassifiedDataSet, bf BestFeatureFunc, iterations uint) (*Decision, error) {
	dtree := &Decision{}
	if ds.Instances == nil || len(ds.Instances) == 0 {
		return nil, errors.New("No instances provided")
	} else if dtree.featureName = bf(ds); dtree.featureName == "" { // No features left
		dtree.outputValue, dtree.isOutput = mostPopularTarget(ds.Instances), true
		return dtree, nil
	} else if iterations == 0 {
		dtree.featureName = ""
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
		ds = ClassifiedDataSet{append([]*Instance{}, ds.Instances...)}
		for i := range ds.Instances {
			ds.Instances[i] = ds.Instances[i].Clone()
		}
		for k, v := range bestFeatureValToInstances {
			var err error
			dtree.nextDecisions[k], err = BoundedTrain(ClassifiedDataSet{Instances: v}, bf, iterations-1)
			if err != nil {
				return nil, errors.New(fmt.Sprint("No instances available to extend tree for feature", dtree.featureName, "with value", k, "this shouldn't be possible"))
			}
		}
		return dtree, nil
	}
}

// Calculates the error the provided decision tree encounters in classifying the provided pre-classified dataset.
func (dtree *Decision) CalculateError(ds ClassifiedDataSet) (float64, error) {
	wrongClassifications := 0.0
	for _, inst := range ds.Instances {
		correctTargetValue := inst.TargetValue
		if err := dtree.Classify(inst); err != nil {
			return 1.0, err
		} else if correctTargetValue != inst.TargetValue {
			wrongClassifications++
		}
		inst.TargetValue = correctTargetValue
	}
	return wrongClassifications / float64(len(ds.Instances)), nil
}

// Attempt to classify a provided instance of data. The classification is set in the instance's TargetValue field.
func (dtree *Decision) Classify(inst *Instance) error {
	if dtree.isOutput {
		inst.TargetValue = dtree.outputValue
		return nil
	} else if thisValue, ok := inst.FeatureValues[dtree.featureName]; ok {
		if nextDecision, ok := dtree.nextDecisions[thisValue]; ok {
			return nextDecision.Classify(inst)
		} else {
			return errors.New(fmt.Sprint("No decision node corresponding to instance value of", thisValue, "for", dtree.featureName))
		}
	} else {
		return errors.New(fmt.Sprint("No decision node for feature ", dtree.featureName))
	}
}

func instancesIdentical(insts []*Instance) bool {
	for i := 1; i < len(insts); i++ {
		if insts[i].TargetValue != insts[i-1].TargetValue {
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

// A BestFeature function that uses information gain.
func BestFeatureInformationGain(ds ClassifiedDataSet) string {
	greatestInfoGain := 0.0
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

	var infoGain = entropy(ds.Instances)
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
	H := 0.0
	for _, count := range targetCounts {
		pI := float64(count) / float64(len(insts))
		H += pI * math.Log2(pI)
	}
	return -H
}
