package id3

import (
	"fmt"
	"testing"
)

func TestCandy(t *testing.T) {
	// Testing candy for "yumminess"
	var testDataset = ClassifiedDataSet{
		[]*Instance{
			{map[string]Feature{"salty": false, "sweet": false}, false}, // Bland
			{map[string]Feature{"salty": true, "sweet": false}, false},  // Disgusting
			{map[string]Feature{"salty": true, "sweet": true}, true},    // Savory
			{map[string]Feature{"salty": false, "sweet": true}, true},   // Sugary
		},
	}
	dtree, err := Train(testDataset, BestFeatureInformationGain)
	if err != nil {
		fmt.Printf("%v\n", err)
	} else {
		printDecisionTree(dtree)
	}
}

func printDecisionTree(dtree *Decision) {
	fmt.Printf("%#v\n", dtree)
	if dtree.isOutput {
		fmt.Printf("Output node: %#v\n", dtree.outputValue)
	} else {
		fmt.Printf("Decision node: %#v\n", dtree.featureName)
		for _, subtree := range dtree.nextDecisions {
			printDecisionTree(subtree)
		}
	}
}