package id3

import (
	"fmt"
	"reflect"
	"sort"
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

//1 Sunny Hot High Weak No
//2 Sunny Hot High Strong No
//3 Overcast Hot High Weak Yes
//4 Rain Mild High Weak Yes
//5 Rain Cool Normal Weak Yes
//6 Rain Cool Normal Strong No
//7 Overcast Cool Normal Strong Yes
//8 Sunny Mild High Weak No
//9 Sunny Cool Normal Weak Yes
//10 Rain Mild Normal Weak Yes
//11 Sunny Mild Normal Strong Yes
//12 Overcast Mild High Strong Yes
//13 Overcast Hot Normal Weak Yes
//14 Rain Mild High Strong No
func TestTennis(t *testing.T) {
	stof := map[string]Feature{
		"sunny":    2,
		"overcast": 1,
		"rain":     0,
		"hot":      2,
		"mild":     1,
		"cool":     0,
		"high":     1,
		"normal":   0,
		"strong":   1,
		"weak":     0,
	}
	stot := map[string]Target{
		"yes": true,
		"no":  false,
	}
	var testDataset = ClassifiedDataSet{
		[]*Instance{
			{map[string]Feature{"outlook": stof["sunny"], "temp": stof["hot"], "humidity": stof["high"], "wind": stof["weak"]}, stot["no"]},
			{map[string]Feature{"outlook": stof["sunny"], "temp": stof["hot"], "humidity": stof["high"], "wind": stof["strong"]}, stot["no"]},
			{map[string]Feature{"outlook": stof["overcast"], "temp": stof["hot"], "humidity": stof["high"], "wind": stof["weak"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["rain"], "temp": stof["mild"], "humidity": stof["high"], "wind": stof["weak"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["rain"], "temp": stof["cool"], "humidity": stof["normal"], "wind": stof["weak"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["rain"], "temp": stof["cool"], "humidity": stof["normal"], "wind": stof["strong"]}, stot["no"]},
			{map[string]Feature{"outlook": stof["overcast"], "temp": stof["cool"], "humidity": stof["normal"], "wind": stof["strong"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["sunny"], "temp": stof["mild"], "humidity": stof["high"], "wind": stof["weak"]}, stot["no"]},
			{map[string]Feature{"outlook": stof["sunny"], "temp": stof["cool"], "humidity": stof["normal"], "wind": stof["weak"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["rain"], "temp": stof["mild"], "humidity": stof["normal"], "wind": stof["weak"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["sunny"], "temp": stof["mild"], "humidity": stof["normal"], "wind": stof["strong"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["overcast"], "temp": stof["mild"], "humidity": stof["high"], "wind": stof["strong"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["overcast"], "temp": stof["hot"], "humidity": stof["normal"], "wind": stof["weak"]}, stot["yes"]},
			{map[string]Feature{"outlook": stof["rain"], "temp": stof["mild"], "humidity": stof["high"], "wind": stof["strong"]}, stot["no"]},
		},
	}
	dtree, err := Train(testDataset, BestFeatureInformationGain)

	var expectedTree = []string{
		`outlook[1] ==> true`,
		`outlook[0] ==> wind[0] ==> true`,
		`outlook[0] ==> wind[1] ==> false`,
		`outlook[2] ==> humidity[1] ==> false`,
		`outlook[2] ==> humidity[0] ==> true`,
	}
	sort.Strings(expectedTree)
	if err != nil {
		t.Error("Encountered tree training error", err)
	} else if treeStr := dtree.String(); !reflect.DeepEqual(treeStr, expectedTree) {
		t.Errorf("Expected %#v got %#v\n", expectedTree, treeStr)
	}
}