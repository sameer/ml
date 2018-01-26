package id3

import (
	"fmt"
	"reflect"
	"testing"
)

func btoFeature(f bool) Feature {
	if f {
		return 1
	} else {
		return 0
	}
}

func btoTarget(t bool) Target {
	if t {
		return true
	} else {
		return false
	}
}

func TestCandy(t *testing.T) {
	// Testing candy for "yumminess"
	var testDataset = ClassifiedDataSet{
		[]*Instance{
			{map[string]Feature{"salty": btoFeature(false), "sweet": btoFeature(false)}, btoTarget(false)}, // Bland
			{map[string]Feature{"salty": btoFeature(true), "sweet": btoFeature(false)}, btoTarget(false)},  // Disgusting
			{map[string]Feature{"salty": btoFeature(true), "sweet": btoFeature(true)}, btoTarget(true)},    // Savory
			{map[string]Feature{"salty": btoFeature(false), "sweet": btoFeature(true)}, btoTarget(true)},   // Sugary
		},
	}

	var expectedTree = &Decision{
		featureName: "sweet",
		nextDecisions: map[Feature]*Decision{
			btoFeature(true): {
				isOutput:    true,
				outputValue: btoTarget(true),
			},
			btoFeature(false): {
				isOutput:    true,
				outputValue: btoTarget(false),
			},
		},
	}

	dtree, err := Train(testDataset, BestFeatureInformationGain)

	if err != nil {
		t.Error("Encountered tree training error", err)
	} else if !reflect.DeepEqual(expectedTree, dtree) {
		t.Error("Expected", expectedTree, "got", dtree)
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
