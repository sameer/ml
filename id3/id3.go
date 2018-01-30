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

// Recursively determines a Decision tree's 'pathways'.
func (dtree *Decision) string(parents []*Decision) []string {
	if dtree.isOutput { // Output nodes actually return a slice of one element, the path to reach them.
		sout := ""
		for i, parent := range parents { // Iterate over parents, building the path
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
		// Add the output node value at the end
		sout += fmt.Sprintf("%#v", dtree.outputValue)
		return []string{sout}
	} else { // Non-output nodes are added to the parents slice that is passed in further
		var sout []string
		if parents == nil {
			parents = make([]*Decision, 0)
		}
		parents = append(parents, dtree)
		for _, subtree := range dtree.nextDecisions { // Append every subtree's output to this output
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

// Creates a duplicate or deep clone of an instance.
func (i *Instance) Clone() *Instance {
	clone := &Instance{}
	clone.TargetValue, clone.FeatureValues = i.TargetValue, make(map[string]Feature, len(i.FeatureValues))
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
	// Infinitely bounded trainng
	return LimitedTrain(ds, bf, ^uint(0))
}

// Allows for training with a specified maximum number of iterations
func LimitedTrain(ds ClassifiedDataSet, bf BestFeatureFunc, iterations uint) (*Decision, error) {
	return limitedTrain(ds, bf, &iterations)
}

func limitedTrain(ds ClassifiedDataSet, bf BestFeatureFunc, iterations *uint) (*Decision, error) {
	dtree := &Decision{} // The decision tree node to return
	if ds.Instances == nil || len(ds.Instances) == 0 { // Can't train with no data
		return nil, errors.New("no instances provided")
	} else if dtree.featureName = bf(ds); dtree.featureName == "" { // No features left
		dtree.outputValue, dtree.isOutput = mostPopularTarget(ds.Instances), true
		return dtree, nil
	} else if *iterations == 0 { // Depth bound has been reached
		dtree.outputValue, dtree.isOutput, dtree.featureName = mostPopularTarget(ds.Instances), true, ""
		return dtree, nil
	} else if instancesIdentical(ds.Instances) { // All instances are the same
		dtree.outputValue, dtree.isOutput = ds.Instances[0].TargetValue, true
		return dtree, nil
	} else { // Make a decision node that will have children

		// Sort instances into buckets of feature value
		bestFeatureValToInstances := make(map[Feature][]*Instance, len(ds.Instances))
		for _, inst := range ds.Instances {
			instances, ok := bestFeatureValToInstances[inst.FeatureValues[dtree.featureName]]
			if !ok {
				instances = make([]*Instance, 0)
			}
			bestFeatureValToInstances[inst.FeatureValues[dtree.featureName]] = append(instances, inst)
		}

		// Clone dataset so features can be removed
		ds = ClassifiedDataSet{append([]*Instance{}, ds.Instances...)}
		for i := range ds.Instances {
			ds.Instances[i] = ds.Instances[i].Clone()
			delete(ds.Instances[i].FeatureValues, dtree.featureName)
		}

		// Create subdecisions
		dtree.nextDecisions = make(map[Feature]*Decision, len(bestFeatureValToInstances))
		for k, v := range bestFeatureValToInstances {
			var err error
			*iterations -= 1
			dtree.nextDecisions[k], err = limitedTrain(ClassifiedDataSet{Instances: v}, bf, iterations)
			if err != nil {
				return nil, errors.New(fmt.Sprint("no instances available to extend tree for feature", dtree.featureName, "with value", k, "this shouldn't be possible"))
			}
		}
		return dtree, nil
	}
}

// Prune a trained Decision tree using the Reduced Error Prune method. A set of labeled instances must be provided
// to prune with.
func (thisTree *Decision) ReducedErrorPrune(validate ClassifiedDataSet) error {
	// Use a stack of Decision nodes and applicable subset of the ClassifiedDataSet
	treeStack, dsStack := []*Decision{thisTree}, [][]*Instance{validate.Instances};
	for ; len(treeStack) > 0; {
		// Pop from the stack
		curTree, curDS := treeStack[len(treeStack)-1], dsStack[len(dsStack)-1]
		treeStack, dsStack = treeStack[:len(treeStack)-1], dsStack[:len(dsStack)-1]

		if curTree.isOutput { // Output nodes have no children, there's no point
			continue
		}

		// Sort instances into buckets of feature value
		featureValueToInsts := make(map[Feature][]*Instance, len(curDS))
		for _, inst := range curDS {
			instances, ok := featureValueToInsts[inst.FeatureValues[curTree.featureName]]
			if !ok {
				instances = make([]*Instance, 0)
			}
			featureValueToInsts[inst.FeatureValues[curTree.featureName]] = append(instances, inst)
		}

		// Iterate over all subtrees, attempting to replace them with output nodes for the most popular instance type.
		// If the error isn't reduced, then the subtree is added to the stack so prune attempts can be done on its own
		// subtrees.
		for featureValue, subTree := range curTree.nextDecisions {
			applicableInstances := featureValueToInsts[featureValue]
			prevError, err := thisTree.CalculateError(validate)
			if err != nil {
				return err
			}
			curTree.nextDecisions[featureValue] = &Decision{isOutput: true, outputValue: mostPopularTarget(applicableInstances)}
			postError, err := thisTree.CalculateError(validate)
			if postError > prevError { // An output decision is bad here, replace with original decision and push to stack
				curTree.nextDecisions[featureValue] = subTree
				treeStack = append(treeStack, subTree)
				dsStack = append(dsStack, applicableInstances)
			}
		}
	}
	return nil
}

// Calculates the error the provided decision tree encounters in classifying the provided pre-classified dataset.
func (dtree *Decision) CalculateError(ds ClassifiedDataSet) (float64, error) {
	wrongClassifications := 0.0
	for _, inst := range ds.Instances { // Classify each instance
		correctTargetValue := inst.TargetValue // Keep track of original value
		if err := dtree.Classify(inst); err != nil {
			return 1.0, err
		} else if correctTargetValue != inst.TargetValue {
			wrongClassifications++
		}
		inst.TargetValue = correctTargetValue // Restore original value
	}
	return wrongClassifications / float64(len(ds.Instances)), nil
}

// Attempt to classify a provided instance of data. The classification is set in the instance's TargetValue field.
func (dtree *Decision) Classify(inst *Instance) error {
	if dtree.isOutput {
		inst.TargetValue = dtree.outputValue // Previous value is overwritten
		return nil
	} else if thisValue, ok := inst.FeatureValues[dtree.featureName]; ok {
		if nextDecision, ok := dtree.nextDecisions[thisValue]; ok {
			return nextDecision.Classify(inst)
		} else {
			return errors.New(fmt.Sprint("no decision node corresponding to instance value of ", thisValue, " for ", dtree.featureName))
		}
	} else {
		return errors.New(fmt.Sprint("no decision node for feature ", dtree.featureName))
	}
}

// Checks if all instances provided have the same target value
func instancesIdentical(insts []*Instance) bool {
	for i := 1; i < len(insts); i++ {
		if insts[i].TargetValue != insts[i-1].TargetValue {
			return false
		}
	}
	return true
}

// Identifies the most 'popular' target value in the slice of instances passed
func mostPopularTarget(insts []*Instance) Target {
	targetCounts := make(map[Target]int, len(insts))
	highestCount := 0
	var highestTarget Target
	for _, inst := range insts {
		count, _ := targetCounts[inst.TargetValue]
		count++
		targetCounts[inst.TargetValue] = count
		if count > highestCount {
			highestCount = count
			highestTarget = inst.TargetValue
		}
	}
	return highestTarget
}

// A BestFeature function that uses information gain to determine the best feature.
func BestFeatureInformationGain(ds ClassifiedDataSet) string {
	greatestInfoGain := 0.0
	greatestFeatureName := ""
	for featureName := range ds.Instances[0].FeatureValues {
		infoGain := infoGainOfFeature(ds, featureName)
		if infoGain > greatestInfoGain { // Determine feature with greatest info gain
			greatestInfoGain = infoGain
			greatestFeatureName = featureName
		}
	}
	return greatestFeatureName
}

var _ BestFeatureFunc = BestFeatureInformationGain

// Determines the information gain of a specified feature for a ClassifiedDataSet.
func infoGainOfFeature(ds ClassifiedDataSet, featureName string) float64 {
	// Count number of each feature value and keep track of the current feature's value for each inst
	featureValueCounts := make(map[Feature]int, len(ds.Instances))
	indexToThisFeature := make([]Feature, len(ds.Instances))
	for i, inst := range ds.Instances {
		thisFeatureValue := inst.FeatureValues[featureName]
		featureValueCounts[thisFeatureValue]++
		indexToThisFeature[i] = thisFeatureValue
	}

	infoGain := entropy(ds.Instances) // Get entropy

	for featureValue, featureCount := range featureValueCounts { // Subtract from entropy to get info gain
		featureValueInsts := make([]*Instance, 0, len(ds.Instances)) // Instances with featureValue
		for i, inst := range ds.Instances {
			if indexToThisFeature[i] == featureValue {
				featureValueInsts = append(featureValueInsts, inst)
			}
		}
		featureValueEntropy := entropy(featureValueInsts) // Entropy of the instances
		infoGain -= float64(featureCount) / float64(len(ds.Instances)) * featureValueEntropy
	}

	return infoGain
}

// Calculates entropy of the targetvalues of a slice of instances.
func entropy(insts []*Instance) float64 {
	targetCounts := make(map[Target]int, len(insts))
	for _, inst := range insts {
		targetCounts[inst.TargetValue]++
	}
	H := 0.0
	for _, count := range targetCounts {
		pI := float64(count) / float64(len(insts))
		H += pI * math.Log2(pI)
	}
	return -H
}
