package id3

import (
	"encoding/csv"
	"math/rand"
	"os"
	"reflect"
	"sort"
	"testing"
	"time"
	"fmt"
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

func TestMushroomEdibility(t *testing.T) {
	indexToFeatureName := []string{
		"",
		"cap-shape",
		"cap-surface",
		"cap-color",
		"bruises?",
		"odor",
		"gill-attachment",
		"gill-spacing",
		"gill-size",
		"gill-color",
		"stalk-shape",
		"stalk-root",
		"stalk-surface-above-ring",
		"stalk-surface-below-ring",
		"stalk-color-above-ring",
		"stalk-color-below-ring",
		"veil-type",
		"veil-color",
		"ring-number",
		"ring-type",
		"spore-print-color",
		"population",
		"habitat",
	}
	featureNameToFeatureValues := map[string]map[string]Feature{}
	for _, feature := range indexToFeatureName {
		featureNameToFeatureValues[feature] = make(map[string]Feature)
	}

	makeDataSet := func(file *os.File) (ClassifiedDataSet, [][]string) {
		r := csv.NewReader(file)
		rows, err := r.ReadAll()
		file.Close()
		if err != nil {
			t.Error(err)
		}
		ds := ClassifiedDataSet{}
		var dsRecords [][]string
		for _, row := range rows {
			inst := &Instance{}
			if row[0] == "p" {
				inst.TargetValue = false
			} else if row[0] == "e" {
				inst.TargetValue = true
			} else {
				t.Error("Invalid value in row")
			}
			inst.FeatureValues = make(map[string]Feature)
			for i := 1; i < len(row); i++ {
				if row[i] == "?" {
					inst = nil
					break
				}
				featureName := indexToFeatureName[i]
				featureValue, ok := featureNameToFeatureValues[featureName][row[i]]
				if !ok {
					featureNameToFeatureValues[featureName][row[i]] = Feature(len(featureNameToFeatureValues[featureName]))
				}
				inst.FeatureValues[featureName] = featureValue
			}
			if inst != nil {
				if len(inst.FeatureValues) != 22 {
					panic("wrong feature length")
				}
				ds.Instances = append(ds.Instances, inst)
				dsRecords = append(dsRecords, row)
			}
		}
		return ds, dsRecords
	}
	for v := 0; v < 1; v++ {
		trainIn, _ := os.Open("train.data")
		testIn, _ := os.Open("test.data")
		validateIn, _ := os.Open("validate.data")
		trainDataSet, _ := makeDataSet(trainIn)
		testDataSet, _ := makeDataSet(testIn)
		validateDataSet, _ := makeDataSet(validateIn)

		for i := 0; i < 50; i++ {
			dtree, err := LimitedTrain(trainDataSet, BestFeatureInformationGain, i)
			if err != nil {
				t.Error(err)
			} else {
				for rep := 0; rep < 2; rep++ {
					if rep == 1 {
						dtree.ReducedErrorPrune(validateDataSet)
					}
					trainError, err := dtree.CalculateError(trainDataSet)
					testError, err := dtree.CalculateError(testDataSet)
					validateError, err := dtree.CalculateError(validateDataSet)
					if err != nil {
						t.Error(err)
					}
					fmt.Print(i, ",", trainError, ",", validateError, ",", testError)
					//fmt.Print(i, ",", trainError, ",", testError)
					if rep == 0 {
						fmt.Print(",")
						//fmt.Println()
					} else {
						fmt.Println()
					}
				}
			}
		}
	}
}

func shuffle(rows [][]string) {
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))
	for a, b := range rng.Perm(len(rows)) {
		rows[a], rows[b] = rows[b], rows[a]
	}
}
